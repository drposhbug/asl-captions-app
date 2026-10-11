"""Import the phrase sheet (Ethan signing seven airport phrases on the Ace Pro 2) into signs/samples.jsonl.

The sheet (docs.google.com/spreadsheets/d/1R-2YBC0gIkextaMGgKtUkKgAUXWYuhrPYixA5E-0hMo) names one trimmed
export per phrase take. The exports are not on disk, but their millisecond stamps are the export times of
Cindy's Camera01 clips (clip start + 4 h clock offset + 20-40 s), so CLIPS maps each camera clip to its phrase.
Every observed-hand frame of a clip is labelled with the phrase's gloss key (what phrases.py speaks), so the
per-frame classifier plus the per-sentence vote in hands_server.camera_loop reads out the whole phrase.

Usage: python import_sheet.py [--out PATH] [--eval] [--dry-run] | --selftest
  --eval  leave-one-clip-out: train on the other clips, majority-vote the held-out clip, print right/wrong.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import signs
from import_cindy import TRACKS, read_track, track_path

MAX_PER_CLIP = 150  # frames kept per clip, spread evenly: the 30 s take must not outweigh the 3 s ones
PER_LABEL = 240  # frames per label after balancing (WHICH GATE THIS had 4 clips, ME LOST 2)
REST = 0.12  # the first and last 12% of a take is the hands rising and dropping: the NOTHING class
CLIPS = {  # camera clip -> gloss key in phrases.PHRASES (sheet column); 015 has no track yet
    "VID_20261010_175752_003": "THIS GATE C4", "VID_20261010_183847_015": "THIS GATE C4",
    "VID_20261010_184035_016": "THIS GATE C4",
    "VID_20261010_175600_001": "WHICH GATE THIS", "VID_20261010_183301_010": "WHICH GATE THIS",
    "VID_20261010_183518_012": "WHICH GATE THIS", "VID_20261010_183354_011": "WHICH GATE THIS",
    "VID_20261010_181016_005": "ME DEAF", "VID_20261010_184209_017": "ME DEAF",
    "VID_20261010_181117_006": "WHERE GATE C4", "VID_20261010_184805_018": "WHERE GATE C4",
    "VID_20261010_181442_007": "THIS FLIGHT DELAYED", "VID_20261010_184941_019": "THIS FLIGHT DELAYED",
    "VID_20261010_182244_009": "WHEN FOOD ARRIVE",
    "VID_20261010_190304_022": "ME LOST", "VID_20261010_190322_023": "ME LOST",
}


def load(tracks=TRACKS, clips=CLIPS):
    """{clip: (label, [hands...])} for every clip with a track, at most MAX_PER_CLIP frames each."""
    out = {}
    for clip, label in clips.items():
        p = track_path(tracks, clip)
        if not p:
            print(f"  NO TRACK  {clip} ({label})")
            continue
        frames = [h for _, h in read_track(p)]
        edge = int(len(frames) * REST)
        rest, frames = frames[:edge] + frames[-edge:], frames[edge:len(frames) - edge]
        step = max(1, len(frames) // MAX_PER_CLIP)
        out[clip] = (label, frames[::step][:MAX_PER_CLIP])
        out[clip + "/rest"] = ("NOTHING", rest[::step])
    return out


def balanced(data, per_label=PER_LABEL):
    """Same frame count for every label: fewer frames from labels with many clips, repeats for labels with few."""
    by_label = {}
    for label, frames in data.values():
        by_label.setdefault(label, []).extend(frames)
    return {label: [frames[int(i)] for i in np.linspace(0, len(frames) - 1, per_label)]
            for label, frames in by_label.items() if frames}


def run(tracks=TRACKS, out=signs.SAMPLES, dry_run=False, clips=CLIPS):
    signs.SAMPLES, signs.DIR = out, os.path.dirname(os.path.abspath(out))
    per_label = Counter()
    for label, frames in balanced(load(tracks, clips)).items():
        for hands in frames:
            if not dry_run:
                signs.add_sample(label, hands)
            per_label[label] += 1
    print(f"{'DRY RUN: would add' if dry_run else 'added'} {sum(per_label.values())} samples -> {out}")
    for label, n in sorted(per_label.items()):
        print(f"  {label:<20} {n}")
    return per_label


def evaluate(tracks=TRACKS, clips=CLIPS):
    """Leave one clip out, majority-vote its frames; the honest number for a new take of a known phrase."""
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    data = load(tracks, clips)
    right = 0
    for held, (label, frames) in data.items():
        if label == "NOTHING":
            continue
        train = balanced({c: v for c, v in data.items() if c != held})
        X = [signs.features(h) for fs in train.values() for h in fs]
        y = [l for l, fs in train.items() for _ in fs]
        m = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=0).fit(np.array(X), y)
        votes = Counter(m.predict(np.array([signs.features(h) for h in frames])))
        votes.pop("NOTHING", None)  # predict() never votes NOTHING
        guess, n = votes.most_common(1)[0]
        right += guess == label
        print(f"  {'ok  ' if guess == label else 'MISS'} {held}  {label:<20} -> {guess:<20} {n}/{len(frames)} frames")
    clips = sum(1 for label, _ in data.values() if label != "NOTHING")
    print(f"leave-one-clip-out: {right}/{clips} clips voted right")
    return right, clips


def selftest():
    import json
    import tempfile
    d = tempfile.mkdtemp()
    pts = [[0.1 * i, 0.2] for i in range(21)]
    with open(os.path.join(d, "clipA_wilor_world_filled.jsonl"), "w") as f:
        for i in range(400):
            f.write(json.dumps({"t": i * 0.1, "hands": [{"label": "Right", "inferred": "observed", "image_points": pts}]}) + "\n")
    got = run(d, os.path.join(d, "signs", "samples.jsonl"), clips={"clipA": "ME LOST", "clipB": "ME DEAF"})
    assert got == {"ME LOST": PER_LABEL, "NOTHING": PER_LABEL}, got  # balanced; clipB has no track
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", default=TRACKS)
    ap.add_argument("--out", default=signs.SAMPLES)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.eval:
        evaluate(a.tracks)
    else:
        run(a.tracks, a.out, a.dry_run)
