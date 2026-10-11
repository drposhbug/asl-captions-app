"""Track the phrase-sheet clips with our own WiLoR in the live camera domain (portrait 360x640, what
hands_server --camera --rotate cw sees from the chest webcam) so the classifier trains on the landmarks it sees live.

Writes signs/tracks/<clip>.jsonl, one line per frame with hands: {"t": s, "hands": [{"handedness", "points"}]}.
Usage: python track_clips.py [--force]   (skips clips whose track exists)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2

import hands_server
from import_sheet import CLIPS

VIDEOS = r"C:\Users\thorn\insta360_backup\Camera01"  # mixed 1080p-4K, 50-200 fps
TRACKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signs", "tracks")
PER_SECOND = 10  # frames tracked per second of clip


def track(pipeline, video, out):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50
    step = max(1, round(fps / PER_SECOND))
    i = kept = 0
    with open(out, "w") as f:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                _, frame = cap.retrieve()
                if frame.shape[1] > frame.shape[0]:  # untagged clips read landscape, face at the left edge; tagged ones already portrait
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame = cv2.resize(frame, (360, 640), interpolation=cv2.INTER_AREA)
                hands = hands_server.detect(pipeline, frame)
                if hands:
                    f.write(json.dumps({"t": round(i / fps, 3), "hands": hands}) + "\n")
                    kept += 1
            i += 1
    cap.release()
    return (i + step - 1) // step, kept


if __name__ == "__main__":
    os.makedirs(TRACKS, exist_ok=True)
    pipeline = hands_server.load_pipeline()
    for clip in CLIPS:
        out = os.path.join(TRACKS, clip + ".jsonl")
        if os.path.exists(out) and "--force" not in sys.argv:
            print(f"skip {clip} (exists)")
            continue
        video = os.path.join(VIDEOS, clip + ".mp4")
        if not os.path.exists(video):
            print(f"NO VIDEO {clip}")
            continue
        t0 = time.time()
        total, kept = track(pipeline, video, out)
        print(f"{clip}: {kept} frames with hands of {total} tracked in {time.time() - t0:.0f}s", flush=True)
