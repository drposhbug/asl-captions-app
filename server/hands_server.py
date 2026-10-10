"""WiLoR hand tracking for the glasses: POST a JPEG to /hands, get 21 normalized landmarks per hand.

WiLoR replaces MediaPipe: it is far more stable on close-up, occluded, head-mounted (egocentric) views.
The glasses runtime has no GPU access for models, so WiLoR runs here and the glasses only draw the result.

With --save-frames, every clean camera frame (no skeleton drawn on it) that contains a hand is saved
under server/frames/<session>/ for uploading to Roboflow and training a sign model on raw images.

Clips recorded on the glasses are POSTed to /clips and saved under server/clips/ for HaWoR (run_hawor.py).

Run:       python hands_server.py [--save-frames]
Self-check: python hands_server.py --selftest
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

import gloss

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
        return WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=torch.float16)
    print("WARNING: no CUDA GPU found; WiLoR on CPU takes seconds per frame. Install the CUDA build of torch.")
    return WiLorHandPose3dEstimationPipeline(device=torch.device("cpu"), dtype=torch.float32)


def decode(jpeg_bytes):
    bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("body is not a decodable image")
    return bgr


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


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/clips":
            return self.save_clip()
        if self.path != "/hands":
            return self.reply(404, {"error": "not found"})
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
            return self.reply(503, {"error": "hand model not installed on the server"})
        hands = detect(self.server.pipeline, bgr)
        self.server.last = (bgr, hands)
        # Sign classification runs on the clean frame only when a hand is in view.
        sign = gloss.predict(body) if hands else None
        if hands and self.server.session_dir:
            # Save the original bytes: the clean frame, exactly as the glasses camera saw it.
            name = f"{time.time():.3f}.jpg"
            with open(os.path.join(self.server.session_dir, name), "wb") as file:
                file.write(body)
        self.reply(200, {"hands": hands, "gloss": sign})

    def save_clip(self):
        extension = CLIP_TYPES.get((self.headers.get("Content-Type") or "").split(";")[0].strip())
        length = int(self.headers.get("Content-Length") or 0)
        if not extension or not 0 < length <= MAX_CLIP:
            return self.reply(400, {"error": "send an mp4 or webm body up to 300 MB"})
        os.makedirs(CLIPS_DIR, exist_ok=True)
        path = os.path.join(CLIPS_DIR, time.strftime("%Y%m%d-%H%M%S") + extension)
        with open(path, "wb") as file:
            file.write(self.rfile.read(length))
        print(f"Saved clip {path}")
        self.reply(200, {"saved": os.path.basename(path)})

    # The last glasses frame with the skeleton drawn on it, for the dashboard (and overlay calibration).
    def do_GET(self):
        if self.path.split("?")[0] != "/preview.jpg" or self.server.last is None:
            return self.reply(404, {"error": "no frame yet"})
        bgr, hands = self.server.last
        frame = bgr.copy()
        height, width = frame.shape[:2]
        for hand in hands:
            points = [(int(x * width), int(y * height)) for x, y in hand["points"]]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, points[a], points[b], (94, 255, 64), 2)
            for point in points:
                cv2.circle(frame, point, 3, (94, 255, 64), -1)
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
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
        server.session_dir = None
        if "--save-frames" in sys.argv:
            server.session_dir = os.path.join(FRAMES_DIR, time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(server.session_dir)
            print(f"Saving clean frames with hands to {server.session_dir}")
        print(f"Hand server on port {PORT}. Set HAND_SERVER_URL in pages/index/index.ink to http://<this computer's LAN IP>:{PORT}/hands")
        server.serve_forever()
