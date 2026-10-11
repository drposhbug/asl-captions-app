"""WiLoR hand tracking for the glasses: POST a JPEG to /hands, get 21 normalized landmarks per hand.

WiLoR replaces MediaPipe: it is far more stable on close-up, occluded, head-mounted (egocentric) views.
The glasses runtime has no GPU access for models, so WiLoR runs here and the glasses only draw the result.

With --save-frames, every clean camera frame (no skeleton drawn on it) that contains a hand is saved
under server/frames/<session>/ for uploading to Roboflow and training a sign model on raw images.

Clips recorded on the glasses are POSTed to /clips and saved under server/clips/ for HaWoR (run_hawor.py).

Signs: a local landmark classifier (signs.py) once signs/model.pkl exists, else Roboflow (gloss.py).
POST /record?label=HELLO&seconds=3 saves the next 3 s of hands as samples, GET /record shows counts,
POST /train retrains signs/model.pkl.

Run:       python hands_server.py [--save-frames] [--camera N|clip.mp4 [--rotate cw|ccw|none] [--mirror]]
           (--camera: a laptop camera replaces the glasses' video; --rotate turns each frame into portrait
           like the chest-clamped Insta360 rig the sign model is trained on, default cw; --mirror flips it)
Self-check: python hands_server.py --selftest
"""
import json
from collections import Counter, deque
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import gloss
import phrases
import signs

# torch + WiLoR are a multi-GB install; without them the server still takes clip uploads.
try:
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
except ImportError as error:
    torch = None
    WILOR_IMPORT_ERROR = error

PORT = int(os.environ.get("HAND_PORT", 8765))  # 8767 when behind forward.py
MAX_BODY = 15 * 1024 * 1024  # full-size photos from the glasses photo camera fallback
MAX_CLIP = 300 * 1024 * 1024
HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(HERE, "frames")
CLIPS_DIR = os.path.join(HERE, "clips")
CLIP_TYPES = {"video/mp4": ".mp4", "video/webm": ".webm"}
# Same 21-point skeleton the glasses draw (pages/index/index.ink).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def load_pipeline():
    if torch is None:
        print(f"WARNING: hand tracking disabled ({WILOR_IMPORT_ERROR}); /clips still works. Install requirements.txt.")
        return None
    if torch.cuda.is_available():
        return WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=torch.float16, verbose=False)
    print("WARNING: no CUDA GPU found; WiLoR on CPU takes seconds per frame. Install the CUDA build of torch.")
    return WiLorHandPose3dEstimationPipeline(device=torch.device("cpu"), dtype=torch.float32, verbose=False)


def decode(jpeg_bytes):
    bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("body is not a decodable image")
    return bgr


SEGMENT_SAMPLES = 6  # frames tracked per video segment, evenly spaced
ROTATE_CODES = {"ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "cw": cv2.ROTATE_90_CLOCKWISE, "none": None}
# The glasses camera records sideways (verified on real clips); rotate pieces upright before tracking.
SEGMENT_ROTATE = ROTATE_CODES[os.environ.get("SEGMENT_ROTATE", "ccw")]


def segment_frames(data, extension):
    """Decodes a video segment and returns up to SEGMENT_SAMPLES evenly spaced upright frames."""
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as file:
        file.write(data)
        path = file.name
    try:
        video = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frames.append(frame)
        video.release()
    finally:
        os.remove(path)
    if len(frames) > SEGMENT_SAMPLES:
        step = len(frames) / SEGMENT_SAMPLES
        frames = [frames[min(len(frames) - 1, int((i + 1) * step) - 1)] for i in range(SEGMENT_SAMPLES)]
    return [cv2.rotate(f, SEGMENT_ROTATE) if SEGMENT_ROTATE is not None else f for f in frames]


def signer_hands(hands):
    """Lock on to the wearer: the chest camera sees their hands largest and nearest, so keep the biggest hand per side,
    drop a second detection of the same hand (wrist within 0.05 of a kept one) and any hand under 40% of the largest
    (someone else's, further away). At most two hands."""
    def size(h):
        p = np.asarray(h["points"]); return float(np.linalg.norm(p[9] - p[0]))
    kept = []
    for h in sorted(hands, key=size, reverse=True):
        wrist = np.asarray(h["points"][0])
        if kept and (size(h) < 0.4 * size(kept[0]) or any(np.linalg.norm(wrist - np.asarray(k["points"][0])) < 0.05 for k in kept)):
            continue
        if any(k["handedness"] == h["handedness"] for k in kept) and len({k["handedness"] for k in kept}) == 2:
            continue
        kept.append(h)
        if len(kept) == 2:
            break
    return kept


def detect(pipeline, bgr):
    height, width = bgr.shape[:2]
    hands = []
    # WiLoR's 21 keypoints use the same order as MediaPipe (wrist, then thumb to pinky), in pixels.
    for out in pipeline.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)):
        keypoints = out["wilor_preds"]["pred_keypoints_2d"][0]
        hands.append({
            "handedness": "Right" if out["is_right"] else "Left",
            "points": [[round(float(x) / width, 4), round(float(y) / height, 4)] for x, y in keypoints],
        })
    return signer_hands(hands)


MODEL_LOCK = threading.Lock()  # the GPU pipeline serves HTTP requests and the webcam thread; one at a time


def process_frame(server, bgr, jpeg):
    """Tracks hands in one clean frame; records training samples, classifies the sign, saves the frame."""
    with MODEL_LOCK:
        hands = detect(server.pipeline, bgr)
    server.last = (bgr, hands)
    server.last_at = time.time()
    label, until = server.recording
    if hands and label and time.time() < until:
        signs.add_sample(label, hands)
    # Local landmark model once trained; else Roboflow on the clean frame, only when a hand is in view.
    if signs.ready():
        sign = signs.predict(hands)
    else:
        sign = gloss.predict(jpeg) if hands else None
    if hands and server.session_dir:
        # Save the clean frame, exactly as the camera saw it.
        name = f"{time.time():.3f}.jpg"
        with open(os.path.join(server.session_dir, name), "wb") as file:
            file.write(jpeg)
    return hands, sign


# --camera: a camera on the laptop (webcam facing the signer) replaces the glasses' video stream for
# hand tracking. No Wi-Fi upload, ~30 fps, ~100 ms latency; the glasses keep captions, display and speech.
COMMIT_FRAMES = 3        # a gloss must hold this many frames (mirrors pages/index/signing.js)
MAX_EDGE = 1280          # long edge fed to WiLoR in camera mode (see the resize in camera_loop)
SENTENCE_GAP_S = 2.0     # hands gone or still for this long ends the sentence (a mid-phrase unsure classifier does not)
STILL_SPEED = 0.6        # wrist speed (frame widths per second) below which a hand counts as still: 97% of real takes hold under it, a hand crossing the frame never does
HOLD_S = 0.3             # a phrase ends with the final handshape held still in view this long; swinging arms never do
MIN_PHRASE_S = 1.5       # first to last confident frame: a single hand movement is shorter, every sheet phrase is longer
# Every sheet phrase moves a wrist at least 0.16 of the frame (measured on the clips); a hand holding a phone
# or resting jitters ~0.02-0.05. Below this the "sentence" is a still hand and nothing is spoken.
MIN_MOTION = 0.10
SURE_SHARE = 0.6         # vote share above which the classifier is trusted as is; below it Claude re-ranks with context
# Keys 1-7 in the live window, in the phrase sheet's column order (how Ethan records them).
# Key 1 is retired with "Is this gate C4" (too close to "Where is gate C4").
KEY_PHRASES = [None, "WHICH GATE THIS", "ME DEAF", "WHERE GATE C4", "THIS FLIGHT DELAYED", "WHEN FOOD ARRIVE", "ME LOST"]
TAKE_S = 30              # safety cap on one training take; the same key pressed again stops it sooner
MIN_LABEL_S = 1.2        # seconds of confident frames in a phrase before it is spoken; a frame count broke when the detection rate climbed
WINNER_S = 0.8           # seconds of confident frames the winning label itself needs; flicker between labels never adds up
RECENCY_FRAMES = 10      # e-folding length of the sentence vote: the end of a sign (C4) decides, not the GATE it shares with WHICH GATE THIS
HUB_URL = os.environ.get("HUB_URL", "http://127.0.0.1:8766")


def wrist_span(frames):
    """Largest distance the wrist of either hand side covered over these frames (normalised image units)."""
    best = 0.0
    for side in ("Left", "Right"):
        w = np.array([h["points"][0] for hands in frames for h in hands if h["handedness"] == side])
        if len(w) > 2:
            best = max(best, float(np.linalg.norm(w.max(0) - w.min(0))))
    return best


def phrase_ok(label_time, winner):
    """Why a finished phrase is not worth speaking, or None: enough confident signing in total, most of it on the winner."""
    if sum(label_time.values()) < MIN_LABEL_S:
        return "too little signing"
    if label_time[winner] < WINNER_S:
        return "no clear winner"
    return None


def recency_votes(frames, k=RECENCY_FRAMES):
    """Confidence vote over [(label, confidence)] where the last frame weighs 1 and each earlier one e^(-1/k) less."""
    votes = Counter()
    for i, (label, confidence) in enumerate(reversed(frames)):
        votes[label] += confidence * np.exp(-i / k)
    return votes


def longest_hold(frames, since):
    """Longest stretch (s) of hands in view and still, over (t, present, moving) records after `since`."""
    best, run_start = 0.0, None
    for t, present, moving in frames:
        if t < since:
            continue
        if present and not moving:
            run_start = t if run_start is None else run_start
            best = max(best, t - run_start)
        else:
            run_start = None
    return best


def draw_hands(frame, hands):
    """The green skeleton over a frame, from normalised points."""
    height, width = frame.shape[:2]
    for hand in hands:
        points = np.array([(int(x * width), int(y * height)) for x, y in hand["points"]], np.int32)
        cv2.polylines(frame, [points[[a, b]] for a, b in HAND_CONNECTIONS], False, (94, 255, 64), 2)


def camera_loop(server, index, rotate="cw", mirror=False):
    import urllib.request
    from urllib.parse import quote

    import polish
    import tts

    # A file path replays a recorded clip through the same pipeline (landmarks, classifier, speech), no camera needed.
    sys.setswitchinterval(0.001)
    live = isinstance(index, int)

    def open_camera():
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # compressed USB stream: fewer stalls
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    cap = open_camera() if live else cv2.VideoCapture(index)
    server.camera_mode = True  # glasses photos would fight the webcam for the GPU and overwrite the skeleton
    while live and not cap.isOpened():  # the camera may still be reconnecting after a replug: wait for it
        print(f"camera {index} not found, retrying in 3 s (replug the camera or check its USB webcam mode)")
        time.sleep(3)
        cap = open_camera()
    if not cap.isOpened():
        print(f"camera {index} not found")
        return
    if not live:
        print(f"clip orientation tag {cap.get(cv2.CAP_PROP_ORIENTATION_META):.0f} deg (applied by OpenCV before --rotate {rotate})")
    print(f"Tracking hands from camera {index} (rotate {rotate}, mirror {'on' if mirror else 'off'}); "
          f"sentences go to the glasses via {HUB_URL}/say and this laptop's speaker. q in the window quits it.")

    def post(path, payload):
        try:
            urllib.request.urlopen(urllib.request.Request(
                HUB_URL + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}), timeout=5)
        except Exception as error:
            print("hub unreachable:", error)

    def say(text, glosses):
        post("/say", {"text": text, "glosses": " ".join(glosses)})
        try:  # Tyler's ElevenLabs voice from the hub's cache, played here so no browser page is needed
            tts.play(urllib.request.urlopen(HUB_URL + "/tts?quiet=1&text=" + quote(text), timeout=20).read())
        except Exception as error:
            print("tts failed:", error)

    def recent_captions(seconds=60, lines=5):
        """What people near the signer said lately, from the hub; [] if it is down."""
        try:
            items = json.load(urllib.request.urlopen(HUB_URL + "/captions", timeout=1))["captions"]
        except Exception:
            return []
        return [c["text"] for c in items if c["at"] > (time.time() - seconds) * 1000][-lines:]

    state = {"glosses": [], "label_time": Counter(), "moves": [], "labels": [], "wrists": {}, "frames": deque(maxlen=150), "said_at": deque(maxlen=8), "candidate": None, "held": 0, "last_activity": 0.0,
             "sign": "", "said": "", "det_fps": 0.0}

    def step(bgr):
        """Landmarks, classifier, gloss commit and the spoken sentence for one frame."""
        jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        hands, sign = process_frame(server, bgr, jpeg)
        state["hands"] = hands
        now = time.time()
        label = sign["label"] if sign else None
        # Activity = a confident label, or a hand that is moving. The phrase ends when hands stop or leave for
        # SENTENCE_GAP_S, not when the classifier merely loses confidence mid-sign.
        wrists = {h["handedness"]: np.asarray(h["points"][0]) for h in hands}
        dt = max(now - state.get("wrists_at", now - 0.05), 0.02)
        moving = any(side in state["wrists"] and np.linalg.norm(w - state["wrists"][side]) / dt > STILL_SPEED
                     for side, w in wrists.items())
        state["wrists"], state["wrists_at"] = wrists, now
        state["frames"].append((now, bool(hands), moving))
        if moving and state["glosses"]:
            state["last_activity"] = now
        if hands and (label or state["labels"]):
            state["moves"].append(hands)  # the whole phrase counts for motion, not only its confident frames
        if label:
            state["last_activity"] = now
            if not state["labels"]:
                state["first_label_at"] = now
            state["label_time"][label] += min(dt, 0.2)  # phrase labels flicker frame to frame: the sentence is the vote
            state["labels"].append((label, sign["confidence"]))
            state["held"] = state["held"] + 1 if label == state["candidate"] else 1
            state["candidate"] = label
            if state["held"] == COMMIT_FRAMES and (not state["glosses"] or state["glosses"][-1] != label):
                state["glosses"].append(label)
                state["sign"] = f"{label}  {sign['confidence']:.2f}"
                print("gloss", label)
                post("/events", {"type": "sign", "text": label, "confidence": sign["confidence"]})  # dashboard readout
        else:
            state["candidate"], state["held"] = None, 0
        if state["glosses"] and now - state["last_activity"] > SENTENCE_GAP_S:
            glosses, votes, label_time = state["glosses"], recency_votes(state["labels"]), state["label_time"]
            motion = wrist_span(state["moves"])
            state["moves"], state["labels"] = [], []
            length = now - state.get("first_label_at", now)
            hold = longest_hold(state["frames"], now - SENTENCE_GAP_S - 3.0)
            # The recency-weighted vote over the sentence wins over the last locked-in sign (sheet phrases are single
            # labels); multi-gloss signing still matches the signer's phrase table, then the language model.
            top = votes.most_common(3)
            why = phrase_ok(label_time, top[0][0]) or ("no motion" if motion < MIN_MOTION else "too short" if length < MIN_PHRASE_S + SENTENCE_GAP_S
                                                       else "no hold at the end" if hold < HOLD_S else None)
            if why:
                state["said"] = f"(dropped: {why})"
                print(f"dropped ({why}; signing {sum(label_time.values()):.1f} s, motion {motion:.2f}, {length - SENTENCE_GAP_S:.1f} s, "
                      f"hold {hold:.1f} s)", {k: round(v, 2) for k, v in label_time.items()})
                state["glosses"], state["label_time"] = [], Counter()
                return
            winner = phrases.match([top[0][0]])
            candidates = [(phrases.match([k]), v / sum(votes.values())) for k, v in top if phrases.match([k])]
            if winner and len(candidates) > 1 and candidates[0][1] < SURE_SHARE and polish.available():
                # Claude re-ranks a close vote using what people nearby just said, minus echoes of our own voice.
                said = {w for line, at in state["said_at"] if now - at < 60 for w in re.findall(r"[a-z']+", line.lower())}
                heard = [c for c in recent_captions()
                         if len(set(re.findall(r"[a-z']+", c.lower())) & said) < 2]
                pick = polish.pick_phrase(candidates, heard)
                print("pick", [(s, round(p, 2)) for s, p in candidates], f"captions {len(heard)}",
                      f"claude {pick!r} vote {winner!r}")
                winner = pick or winner
            text = winner or phrases.match(glosses) or (
                polish.sentence_from_glosses(glosses, []) if polish.available() else " ".join(glosses).capitalize() + ".")
            print("sentence", glosses, {k: round(v, 1) for k, v in votes.items()}, f"motion {motion:.2f} ->", text)
            state["said"] = text
            state["said_at"].append((text, now))
            say(text, glosses)
            state["glosses"], state["label_time"] = [], Counter()
        elif not state["glosses"] and now - state["last_activity"] > SENTENCE_GAP_S:
            state["label_time"], state["moves"], state["labels"] = Counter(), [], []  # guesses that never locked in do not carry into the next phrase

    if polish.available():  # the first Claude call per process takes ~4 s (connection setup): pay it now, not on the first phrase
        threading.Thread(target=polish.pick_phrase, args=([("I'm deaf.", 1.0)], []), daemon=True).start()

    pending = deque(maxlen=1)  # the newest camera frame: WiLoR (~0.2 s) takes whichever is current, the display never waits

    def detector():
        while True:
            if not pending:
                time.sleep(0.005)
                continue
            t0 = time.time()
            try:
                step(pending.pop())
            except Exception:  # one bad frame or a failed Claude call must not kill the only detector thread
                traceback.print_exc()
                state["glosses"], state["label_time"], state["moves"], state["labels"] = [], Counter(), [], []
            state["det_fps"] = 1 / max(time.time() - t0, 1e-3)

    if live:
        threading.Thread(target=detector, daemon=True).start()
    def retrain():
        """Rebuild the model from every sample in a thread; the detector hot-reloads model.pkl when it lands."""
        if state.get("training"):
            return
        def train():
            try:
                result = signs.train()
                state["trained"] = f"trained {result['samples']} samples, held-out {result['accuracy']:.0%}"
            except Exception as error:
                state["trained"] = f"train failed: {error}"
            print(state["trained"])
            state["training"] = False
        state["training"], state["trained"] = True, "training..."
        threading.Thread(target=train, daemon=True).start()

    window = "ASL Captions - sign to speech"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # resizable, kept smaller than a 1080p laptop screen
    cv2.resizeWindow(window, 405, 720)
    shown_n, shown_t, stalls = 0, time.time(), 0
    while True:
        read_at = time.time()
        if live and server.last_at and read_at - server.last_at > 30:
            # Watchdog: the camera handle can wedge after a USB stall and never recover; the launcher restarts us.
            print("no camera frame for 30 s: exiting so the launcher restarts the hand server")
            os._exit(3)
        ok, bgr = cap.read()
        if live and time.time() - read_at > 0.2:
            stalls += 1
            print(f"camera stalled {time.time() - read_at:.2f} s (ok={ok}): USB stream paused, not the model")
            if stalls >= 5:  # the Ace Pro 2 webcam stream sometimes drops to 1 fps (green frames) until it is reopened
                print("reopening the camera")
                cap.release()
                time.sleep(1)
                cap = open_camera()
                stalls = 0
                continue
            # a slow frame is still a frame: keep tracking at 1 fps rather than showing nothing
        else:
            stalls = 0
        shown_n += 1
        if time.time() - shown_t >= 5:  # display rate, for lag debugging
            state["fps"] = shown_n / (time.time() - shown_t)
            print(f"display {state['fps']:.1f} fps, {state['det_fps']:.1f} detections/s")
            shown_n, shown_t = 0, time.time()
        if not ok:
            if not live:
                print("clip finished")
                break
            stalls += 1
            if stalls >= 20:  # the camera went away (unplugged / left webcam mode): reopen until it is back
                print("camera read failed 20 times: reopening (replug the camera if this repeats)")
                cap.release()
                time.sleep(2)
                cap = open_camera()
                stalls = 0
            time.sleep(0.1)
            continue
        # Match the training clips' geometry (camera clamped on its side on the chest, frames turned to
        # portrait): the landmark classifier is not rotation invariant.
        # "auto" (replays): the clips are mixed, some carry an orientation tag OpenCV applies (already portrait), others
        # come out landscape with the wearer's face at the left edge and need a clockwise turn.
        code = ROTATE_CODES["cw"] if rotate == "auto" and bgr.shape[1] > bgr.shape[0] else ROTATE_CODES.get(rotate)
        if code is not None:
            bgr = cv2.rotate(bgr, code)
        if mirror:
            bgr = cv2.flip(bgr, 1)
        # WiLoR cost scales with pixels (measured: 3.8 s at 4K, 0.6 s at 1080p, 0.28 s at 720p for two hands on the
        # RTX 3050), so cap the long edge.
        h, w = bgr.shape[:2]
        if max(h, w) > MAX_EDGE:
            scale = MAX_EDGE / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if live:
            pending.append(bgr)
        else:
            step(bgr)  # a replay processes every frame
        if window is None:
            continue
        shown = bgr.copy()
        if server.last is not None:
            draw_hands(shown, server.last[1])
        bottom = shown.shape[0]
        hands = state.get("hands") or []
        rec_label, rec_until = server.recording
        recording = time.time() < rec_until
        if state.get("was_recording") and not recording:
            retrain()  # every finished take trains in the background, so the model keeps up while takes continue
        state["was_recording"] = recording
        lines = [f"1 camera  {state.get('fps', 0):.0f} fps",
                 f"2 hands   {len(hands)} " + " ".join(h["handedness"][0] for h in hands) + f"  {state['det_fps']:.0f}/s",
                 "3 model   " + ("  ".join(f"{l[:10]} {p:.0%}" for l, p in signs.latest["top"][:2]) or "-"),
                 "4 sign    " + (state["sign"] or "-"),
                 "5 said    " + (state["said"] or "-")]
        if time.time() < rec_until:
            lines.append(f"REC {rec_label}  {time.time() - state['take_at'] / 1000:.0f} s  press the key again to stop")
        elif state.get("trained"):
            lines.append(state["trained"])
        else:
            lines.append("keys: 2-7 start/stop a phrase take, 0 nothing, u undo take (then t), t train, q quit")
        for i, line in enumerate(lines):
            if line:
                origin = (8, 22 + 20 * i)  # top-left: the bottom of a portrait frame falls off a laptop screen
                cv2.putText(shown, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(shown, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (94, 255, 64), 1, cv2.LINE_AA)
        try:
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # Record training takes on this very camera: 1-7 = the sheet phrases (sheet column order), 0 = NOTHING
            # (hands at rest / holding a phone), the same key again stops the take, u = undo the last take, t = retrain.
            # Order is the sheet's columns: 1 (retired), 2 Which gate did they say, 3 I'm deaf, 4 Where is gate C4,
            # 5 Is this flight delayed, 6 When does the food arrive, 7 I'm lost.
            if key in range(ord("0"), ord("8")):
                label = "NOTHING" if key == ord("0") else KEY_PHRASES[key - ord("1")]
                if label is None:
                    state["trained"] = "key 1 is retired (Is this gate C4 was removed)"
                    continue
                rec_label, rec_until = server.recording
                if rec_label == label and time.time() < rec_until:
                    server.recording = (None, 0)
                    print("stopped", label)
                else:
                    state["take_at"], state["take_label"] = int(time.time() * 1000), label
                    server.recording = (label, time.time() + TAKE_S)
                    print("recording", label)
            elif key == ord("u") and state.get("take_at"):
                gone = signs.drop_since(state["take_at"])
                server.recording = (None, 0)
                state["trained"] = f"undid {state['take_label']}: {gone} frames removed"
                state["take_at"] = None
                print(state["trained"])
            elif key == ord("t"):
                retrain()
        except cv2.error as error:  # no display: keep tracking and speaking without the window
            print("no window:", error)
            window = None
    cap.release()
    cv2.destroyAllWindows()


class Handler(BaseHTTPRequestHandler):
    def handle(self):
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # the dashboard poll gave up before the reply was written: not an error worth a traceback

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/clips":
            return self.save_clip()
        if url.path == "/segment":
            return self.process_segment()
        if url.path == "/record":
            q = parse_qs(url.query)
            label = q.get("label", [""])[0].strip().upper()
            try:
                seconds = float(q.get("seconds", ["3"])[0])
            except ValueError:
                seconds = 0
            if not re.fullmatch(r"[A-Z0-9_-]{1,32}", label) or not 0 < seconds <= 30:
                return self.reply(400, {"error": "need label=[A-Z0-9_-]{1,32} and seconds in (0, 30]"})
            self.server.recording = (label, time.time() + seconds)
            return self.reply(200, {"ok": True})
        if url.path == "/train":
            try:
                return self.reply(200, signs.train())
            except ValueError as error:
                return self.reply(400, {"error": str(error)})
        # Some clients (the glasses) send an absolute URL as the request path; compare the path part only.
        if url.path != "/hands":
            return self.reply(404, {"error": "not found"})
        if getattr(self.server, "camera_mode", False):
            return self.reply(409, {"error": "hand server is tracking the laptop camera"})
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            return self.reply(400, {"error": "send a JPEG body up to 15 MB"})
        body = self.rfile.read(length)
        try:
            bgr = decode(body)
        except ValueError as error:
            return self.reply(400, {"error": str(error)})
        if self.server.pipeline is None:
            self.server.last = (bgr, [])  # the dashboard preview still shows the raw glasses feed
            self.server.last_at = time.time()
            return self.reply(503, {"error": "hand model not installed on the server"})
        hands, sign = self.process_frame(bgr, body)
        self.reply(200, {"hands": hands, "gloss": sign})

    def process_frame(self, bgr, jpeg):
        return process_frame(self.server, bgr, jpeg)

    def process_segment(self):
        """A 1-3 s video piece from the glasses: far smaller than photos, so captions keep their bandwidth."""
        extension = CLIP_TYPES.get((self.headers.get("Content-Type") or "").split(";")[0].strip())
        length = int(self.headers.get("Content-Length") or 0)
        if not extension or not 0 < length <= MAX_BODY:
            return self.reply(400, {"error": "send an mp4 or webm segment up to 15 MB"})
        data = self.rfile.read(length)
        if len(data) != length:
            return self.reply(400, {"error": "segment upload cut off"})
        frames = segment_frames(data, extension)
        if not frames:
            return self.reply(400, {"error": "segment has no decodable frames"})
        if self.server.pipeline is None:
            self.server.last = (frames[-1], [])
            self.server.last_at = time.time()
            return self.reply(503, {"error": "hand model not installed on the server"})
        hands, sign, per_frame = [], None, []
        for bgr in frames:
            hands, frame_sign = self.process_frame(bgr, cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
            per_frame.append(frame_sign)
            sign = frame_sign or sign
        # "glosses" keeps one entry per tracked frame so the glasses can debounce signs as if frames were live.
        self.reply(200, {"hands": hands, "gloss": sign, "glosses": per_frame, "frames": len(frames)})

    def save_clip(self):
        extension = CLIP_TYPES.get((self.headers.get("Content-Type") or "").split(";")[0].strip())
        length = int(self.headers.get("Content-Length") or 0)
        if not extension or not 0 < length <= MAX_CLIP:
            return self.reply(400, {"error": "send an mp4 or webm body up to 300 MB"})
        data = self.rfile.read(length)
        if len(data) != length:  # the sender gave up mid-upload; don't keep a truncated clip
            return self.reply(400, {"error": f"upload cut off at {len(data)} of {length} bytes"})
        os.makedirs(CLIPS_DIR, exist_ok=True)
        path = os.path.join(CLIPS_DIR, time.strftime("%Y%m%d-%H%M%S") + extension)
        with open(path, "wb") as file:
            file.write(data)
        print(f"Saved clip {path}")
        self.reply(200, {"saved": os.path.basename(path)})

    # The last glasses frame with the skeleton drawn on it, for the dashboard (and overlay calibration).
    def do_GET(self):
        if urlparse(self.path).path == "/record":
            label, until = self.server.recording
            return self.reply(200, {"recording": label if time.time() < until else None, "counts": signs.counts()})
        path = urlparse(self.path).path
        if path not in ("/preview.jpg", "/frame.jpg") or self.server.last is None:
            return self.reply(404, {"error": "no frame yet"})
        bgr, hands = self.server.last
        frame = bgr.copy()
        if path == "/frame.jpg":  # the clean frame (no skeleton), for scene context
            hands = []
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5) if max(frame.shape[:2]) > 1000 else frame
        draw_hands(frame, hands)
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Age", f"{time.time() - self.server.last_at:.1f}")
        self.end_headers()
        self.wfile.write(jpeg.tobytes())

    def reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # one line per frame would flood the console


def selftest():
    held = [(t * 0.1, True, False) for t in range(10)]            # 0.9 s still in view
    swing = [(t * 0.1, True, True) for t in range(10)] + [(1.0 + t * 0.1, False, False) for t in range(5)]
    assert abs(longest_hold(held, 0) - 0.9) < 1e-6 and longest_hold(swing, 0) == 0.0, (longest_hold(held, 0), longest_hold(swing, 0))
    assert longest_hold(held, 0.5) < 0.5, "only records after `since` count"
    def hand(side, x, y, scale):
        return {"handedness": side, "points": [[x, y]] + [[x, y + scale]] * 8 + [[x, y + scale]] * 12}
    big_r, big_l = hand("Right", 0.3, 0.5, 0.15), hand("Left", 0.7, 0.5, 0.14)
    dup = hand("Right", 0.31, 0.51, 0.13)          # same hand detected twice
    far = hand("Left", 0.5, 0.2, 0.04)              # someone else's hand, further away
    got = signer_hands([far, dup, big_l, big_r])
    assert got == [big_r, big_l], got
    assert signer_hands([far]) == [far], "alone, a small hand is still the signer's"
    still = [[{"handedness": "Right", "points": [[0.5 + 0.01 * (i % 2), 0.5]] * 21}] for i in range(10)]
    moving = [[{"handedness": "Right", "points": [[0.2 + 0.05 * i, 0.5]] * 21}] for i in range(10)]
    assert wrist_span(still) < MIN_MOTION < wrist_span(moving), (wrist_span(still), wrist_span(moving))
    # WHERE GATE C4 signed live: a short WHERE, a long GATE the model calls WHICH GATE THIS, then C4; the end decides
    frames = [("WHERE GATE C4", 0.9)] * 3 + [("WHICH GATE THIS", 0.9)] * 20 + [("WHERE GATE C4", 0.9)] * 8
    plain = Counter()
    for label, confidence in frames:
        plain[label] += confidence
    assert plain.most_common(1)[0][0] == "WHICH GATE THIS" and recency_votes(frames).most_common(1)[0][0] == "WHERE GATE C4", recency_votes(frames)
    assert recency_votes([]) == Counter() and abs(recency_votes([("A", 1.0)])["A"] - 1.0) < 1e-9
    # a 2 s take with one dominant label speaks; a 0.3 s flicker burst or a 50/50 split between labels does not
    assert phrase_ok(Counter({"WHERE GATE C4": 1.5, "WHEN FOOD ARRIVE": 0.5}), "WHERE GATE C4") is None
    assert phrase_ok(Counter({"WHERE GATE C4": 0.2, "WHEN FOOD ARRIVE": 0.1}), "WHERE GATE C4") == "too little signing"
    assert phrase_ok(Counter({"WHERE GATE C4": 0.7, "WHEN FOOD ARRIVE": 0.7}), "WHERE GATE C4") == "no clear winner"
    pipeline = load_pipeline()
    _, blank = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))
    assert detect(pipeline, decode(blank.tobytes())) == [], "blank frame should have no hands"
    try:
        decode(b"not an image")
        raise AssertionError("garbage body should be rejected")
    except ValueError:
        pass
    print("selftest ok")


class Tee:
    """stdout that also appends to a file, flushed per line (the launcher window's own buffer never flushes)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        sys.stdout = sys.stderr = Tee(sys.stdout, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hands.log"), "a", encoding="utf-8"))
        # Single-threaded on purpose: one GPU model, one frame at a time.
        server = HTTPServer(("0.0.0.0", PORT), Handler)
        server.pipeline = load_pipeline()
        server.last = None
        server.last_at = 0
        server.recording = (None, 0)  # (label, until epoch s) for POST /record
        server.session_dir = None
        if "--save-frames" in sys.argv:
            server.session_dir = os.path.join(FRAMES_DIR, time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(server.session_dir, exist_ok=True)
            print(f"Saving clean frames with hands to {server.session_dir}")
        if "--camera" in sys.argv:
            arg = sys.argv[sys.argv.index("--camera") + 1]
            index = int(arg) if arg.isdigit() else arg  # a camera index, or a clip path to replay
            # A webcam frame is landscape and needs turning; a replay turns only frames that come out landscape.
            rotate = sys.argv[sys.argv.index("--rotate") + 1] if "--rotate" in sys.argv else ("cw" if arg.isdigit() else "auto")
            if rotate not in ROTATE_CODES and rotate != "auto":
                sys.exit(f"--rotate must be one of {', '.join(ROTATE_CODES)}, auto")
            threading.Thread(target=camera_loop, args=(server, index, rotate, "--mirror" in sys.argv), daemon=True).start()
        print(f"Hand server on port {PORT}. Set HAND_SERVER_URL in pages/index/index.ink to http://<this computer's LAN IP>:{PORT}/hands")
        server.serve_forever()
