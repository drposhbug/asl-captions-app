"""Which signs get mixed up? Classify a folder of labelled frames with the hosted Roboflow model.

Usage: python confusion.py <folder> [--limit 30] [--csv out.csv] | --selftest
<folder> holds one subfolder per true class (frames/HELLO/*.jpg), or a Roboflow classification
export with train/valid/test (valid is used). Needs ROBOFLOW_API_KEY and ROBOFLOW_MODEL, as gloss.py.
"""
import argparse
import csv
import sys
import tempfile
from collections import Counter
from pathlib import Path

from gloss import enabled, predict

EXTS = {".jpg", ".jpeg", ".png"}


def collect(folder, limit):
    root = Path(folder)
    if (root / "valid").is_dir():
        root = root / "valid"
    return {d.name: sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)[:limit]
            for d in sorted(root.iterdir()) if d.is_dir()}


def evaluate(images):
    counts = {}
    for true, paths in images.items():
        counts[true] = Counter()
        for path in paths:
            result = predict(path.read_bytes())
            counts[true][result["label"] if result else "(none)"] += 1
    return counts


def confused_pairs(counts, top=5):
    pairs = Counter()
    for true, row in counts.items():
        for pred, n in row.items():
            if pred != true and pred in counts:
                pairs[tuple(sorted((true, pred)))] += n
    return pairs.most_common(top)


def report(counts, csv_path=None):
    labels = list(counts) + sorted({p for row in counts.values() for p in row} - set(counts))
    print("Per-class accuracy:")
    for true, row in counts.items():
        total = sum(row.values())
        print(f"  {true:<14} {row[true]:>3}/{total:<3} {100 * row[true] / total if total else 0:5.1f}%")
    width = max(6, *(len(l) for l in labels)) + 1
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(" " * width + "".join(f"{l:>{width}}" for l in labels))
    for true, row in counts.items():
        print(f"{true:<{width}}" + "".join(f"{row[l]:>{width}}" for l in labels))
    print("\nMost confused pairs:")
    pairs = confused_pairs(counts)
    for (a, b), n in pairs:
        print(f"  {a} <-> {b}: {n} mistakes -- record more reps of {a} and {b}, "
              "or swap one for a more distinct sign in the demo script")
    if not pairs:
        print("  none")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true/predicted", *labels])
            writer.writerows([true, *(row[l] for l in labels)] for true, row in counts.items())
        print(f"\nWrote {csv_path}")
    return pairs


def selftest():
    global predict
    # fake model: file name says what it "predicts"; content is ignored, so stub bytes suffice
    predict = lambda data: {"label": data.decode(), "confidence": 0.9} if data else None
    plan = {"HELLO": ["HELLO"] * 4 + ["THANKS"] * 2, "THANKS": ["THANKS"] * 3 + ["HELLO"] * 1,
            "COFFEE": ["COFFEE"] * 4 + ["TEA"] * 1, "TEA": ["TEA"] * 4 + [""]}
    with tempfile.TemporaryDirectory() as tmp:
        for cls, preds in plan.items():
            (Path(tmp) / "valid" / cls).mkdir(parents=True)
            for i, p in enumerate(preds):
                (Path(tmp) / "valid" / cls / f"{i}.jpg").write_bytes(p.encode())
        (Path(tmp) / "train" / "IGNORED").mkdir(parents=True)
        counts = evaluate(collect(tmp, limit=30))
        pairs = report(counts, csv_path=str(Path(tmp) / "out.csv"))
        assert "IGNORED" not in counts
        assert counts["HELLO"] == Counter(HELLO=4, THANKS=2)
        assert counts["TEA"]["(none)"] == 1
        assert pairs == [(("HELLO", "THANKS"), 3), (("COFFEE", "TEA"), 1)], pairs
        assert sum(map(len, collect(tmp, limit=2).values())) == 8
        assert (Path(tmp) / "out.csv").read_text().startswith("true/predicted,COFFEE")
    print("\nselftest ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?")
    parser.add_argument("--limit", type=int, default=30, help="images per class (caps API calls)")
    parser.add_argument("--csv")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif not args.folder:
        parser.error("folder required")
    else:
        if not enabled():
            sys.exit("Set ROBOFLOW_API_KEY and ROBOFLOW_MODEL first.")
        report(evaluate(collect(args.folder, args.limit)), args.csv)
