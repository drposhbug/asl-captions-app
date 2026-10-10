"""Process a clip recorded on the glasses: clean frames for Roboflow, then HaWoR hand reconstruction.

HaWoR works on whole videos offline (world-space, temporally consistent), so it runs here, not live.
It has heavy, conflicting dependencies, so it lives in its own checkout and Python environment:
  git clone --recursive https://github.com/ThunderVVV/HaWoR  and follow its README (conda env, weights, MANO).
Then point this script at it with HAWOR_DIR and HAWOR_PYTHON (that env's python.exe).

Run:        python run_hawor.py clips/<clip>.mp4 [--every 5] [--frames-only]
Self-check: python run_hawor.py --selftest
"""
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def extract_frames(clip_path, out_dir, every):
    """Writes every Nth frame as a clean JPEG (nothing drawn on it) and returns how many it saved."""
    video = cv2.VideoCapture(clip_path)
    if not video.isOpened():
        raise ValueError(f"cannot open {clip_path}")
    os.makedirs(out_dir, exist_ok=True)
    index = saved = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if index % every == 0:
            cv2.imwrite(os.path.join(out_dir, f"{index:06d}.jpg"), frame)
            saved += 1
        index += 1
    video.release()
    return saved


def as_mp4(clip_path):
    """HaWoR's demo reads mp4; the glasses may record webm, so re-encode those."""
    if clip_path.lower().endswith(".mp4"):
        return clip_path
    video = cv2.VideoCapture(clip_path)
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    mp4_path = os.path.splitext(clip_path)[0] + ".mp4"
    writer = None
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        writer.write(frame)
    video.release()
    if writer is None:
        raise ValueError(f"no frames in {clip_path}")
    writer.release()
    return mp4_path


def run_hawor(mp4_path):
    hawor_dir, hawor_python = os.environ.get("HAWOR_DIR"), os.environ.get("HAWOR_PYTHON")
    if not hawor_dir or not hawor_python:
        sys.exit("Set HAWOR_DIR (HaWoR checkout) and HAWOR_PYTHON (its env's python) to run HaWoR.")
    # HaWoR writes next to the input video, in <video dir>/<video name>/: extracted_images/, tracks_*/,
    # SLAM/, world_space_res.pth (hand poses) and vis_0_N/aitviewer/video_0.mp4 (the render).
    # demo.py shells out to `ffmpeg`; the conda env's Library\bin holds it but is not on PATH.
    env = dict(os.environ, PATH=os.path.join(os.path.dirname(hawor_python), "Library", "bin") + os.pathsep + os.environ["PATH"])
    subprocess.run(
        [hawor_python, "demo.py", "--video_path", os.path.abspath(mp4_path), "--vis_mode", "world"],
        cwd=hawor_dir,
        env=env,
        check=True,
    )


def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "test.avi")
        writer = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        for _ in range(10):
            writer.write(np.zeros((48, 64, 3), np.uint8))
        writer.release()
        assert extract_frames(clip, os.path.join(tmp, "frames"), every=5) == 2, "every 5th of 10 frames is 2"
        assert os.path.exists(as_mp4(clip)), "non-mp4 clips should be re-encoded"
    print("selftest ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--selftest" in args:
        selftest()
        sys.exit()
    if not args or args[0].startswith("--"):
        sys.exit(__doc__)
    clip = args[0]
    every = int(args[args.index("--every") + 1]) if "--every" in args else 5
    name = os.path.splitext(os.path.basename(clip))[0]
    frames_dir = os.path.join(HERE, "frames", name)
    print(f"Saved {extract_frames(clip, frames_dir, every)} clean frames for Roboflow to {frames_dir}")
    if "--frames-only" not in args:
        run_hawor(as_mp4(clip))
