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
    return hands


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
SENTENCE_GAP_S = 1.5     # no gloss for this long ends the sentence
MIN_VOTES = 6            # summed confidence (about 1.5 s of confident frames) before a sentence is spoken; a hand at rest never reaches it
HUB_URL = os.environ.get("HUB_URL", "http://127.0.0.1:8766")


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
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if live else cv2.VideoCapture(index)
    if live:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # compressed USB stream: fewer stalls
        cap.set(cv2.CAP_PROP_FPS, 30)
    server.camera_mode = True  # glasses photos would fight the webcam for the GPU and overwrite the skeleton
    if not cap.isOpened():
        print(f"camera {index} not found")
        return
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

    state = {"glosses": [], "votes": Counter(), "candidate": None, "held": 0, "last_activity": 0.0,
             "sign": "", "said": "", "det_fps": 0.0}

    def step(bgr):
        """Landmarks, classifier, gloss commit and the spoken sentence for one frame."""
        jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        hands, sign = process_frame(server, bgr, jpeg)
        state["hands"] = hands
        now = time.time()
        label = sign["label"] if sign else None
        if label:
            state["last_activity"] = now
            state["votes"][label] += sign["confidence"]  # phrase labels flicker frame to frame: the sentence is the vote
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
            glosses, votes = state["glosses"], state["votes"]
            if sum(votes.values()) < MIN_VOTES:
                print("too short, dropped", glosses, dict(votes))
                state["glosses"], state["votes"] = [], Counter()
                return
            # The signer's own phrases skip the model, as fillSentence does on the glasses.
            text = phrases.match(glosses) or phrases.match([votes.most_common(1)[0][0]]) or (
                polish.sentence_from_glosses(glosses, []) if polish.available() else " ".join(glosses).capitalize() + ".")
            print("sentence", glosses, dict(votes), "->", text)
            state["said"] = text
            say(text, glosses)
            state["glosses"], state["votes"] = [], Counter()

    pending = deque(maxlen=1)  # the newest camera frame: WiLoR (~0.2 s) takes whichever is current, the display never waits

    def detector():
        while True:
            if not pending:
                time.sleep(0.005)
                continue
            t0 = time.time()
            step(pending.pop())
            state["det_fps"] = 1 / max(time.time() - t0, 1e-3)

    if live:
        threading.Thread(target=detector, daemon=True).start()
    window = "ASL Captions - sign to speech"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # resizable, kept smaller than a 1080p laptop screen
    cv2.resizeWindow(window, 405, 720)
    shown_n, shown_t = 0, time.time()
    while True:
        read_at = time.time()
        ok, bgr = cap.read()
        if live and time.time() - read_at > 0.2:
            print(f"camera stalled {time.time() - read_at:.2f} s (ok={ok}): USB stream paused, not the model")
        shown_n += 1
        if time.time() - shown_t >= 5:  # display rate, for lag debugging
            state["fps"] = shown_n / (time.time() - shown_t)
            print(f"display {state['fps']:.1f} fps, {state['det_fps']:.1f} detections/s")
            shown_n, shown_t = 0, time.time()
        if not ok:
            if not live:
                print("clip finished")
                break
            time.sleep(0.1)
            continue
        # Match the training clips' geometry (camera clamped on its side on the chest, frames turned to
        # portrait): the landmark classifier is not rotation invariant.
        if ROTATE_CODES[rotate] is not None:
            bgr = cv2.rotate(bgr, ROTATE_CODES[rotate])
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
        lines = [f"1 camera  {state.get('fps', 0):.0f} fps",
                 f"2 hands   {len(hands)} " + " ".join(h["handedness"][0] for h in hands) + f"  {state['det_fps']:.0f}/s",
                 "3 model   " + ("  ".join(f"{l[:10]} {p:.0%}" for l, p in signs.latest["top"][:2]) or "-"),
                 "4 sign    " + (state["sign"] or "-"),
                 "5 said    " + (state["said"] or "-")]
        for i, line in enumerate(lines):
            if line:
                origin = (8, 22 + 20 * i)  # top-left: the bottom of a portrait frame falls off a laptop screen
                cv2.putText(shown, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(shown, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (94, 255, 64), 1, cv2.LINE_AA)
        try:
            cv2.imshow(window, shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except cv2.error as error:  # no display: keep tracking and speaking without the window
            print("no window:", error)
            window = None
    cap.release()
    cv2.destroyAllWindows()


class Handler(BaseHTTPRequestHandler):
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
    pipeline = load_pipeline()
    _, blank = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))
    assert detect(pipeline, decode(blank.tobytes())) == [], "blank frame should have no hands"
    try:
        decode(b"not an image")
        raise AssertionError("garbage body should be rejected")
    except ValueError:
        pass
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        # Single-threaded on purpose: one GPU model, one frame at a time.
        server = HTTPServer(("0.0.0.0", PORT), Handler)
        server.pipeline = load_pipeline()
        server.last = None
        server.last_at = 0
        server.recording = (None, 0)  # (label, until epoch s) for POST /record
        server.session_dir = None
        if "--save-frames" in sys.argv:
            server.session_dir = os.path.join(FRAMES_DIR, time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(server.session_dir)
            print(f"Saving clean frames with hands to {server.session_dir}")
        if "--camera" in sys.argv:
            arg = sys.argv[sys.argv.index("--camera") + 1]
            index = int(arg) if arg.isdigit() else arg  # a camera index, or a clip path to replay
            rotate = sys.argv[sys.argv.index("--rotate") + 1] if "--rotate" in sys.argv else "cw"
            if rotate not in ROTATE_CODES:
                sys.exit(f"--rotate must be one of {', '.join(ROTATE_CODES)}")
            threading.Thread(target=camera_loop, args=(server, index, rotate, "--mirror" in sys.argv), daemon=True).start()
        print(f"Hand server on port {PORT}. Set HAND_SERVER_URL in pages/index/index.ink to http://<this computer's LAN IP>:{PORT}/hands")
        server.serve_forever()
