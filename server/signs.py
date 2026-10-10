"""Local sign classifier on WiLoR hand landmarks (replaces the never-trained Roboflow call).

Samples are JSON lines in signs/samples.jsonl ({"label", "hands", "at"}), recorded through the hand
server's POST /record. Train an MLP (KNN while any class has < 5 samples) into signs/model.pkl.

Usage: python signs.py train | --selftest
"""
import json
import os
import pickle
import sys
import time
from collections import Counter, deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "signs")
SAMPLES = os.path.join(DIR, "samples.jsonl")
MODEL = os.path.join(DIR, "model.pkl")
WINDOW, MIN_CONFIDENCE = 5, 0.6

window = deque(maxlen=WINDOW)  # last frames' class probabilities
loaded = {"mtime": None, "model": None}


def hand_vector(points, mirror):
    p = np.asarray(points, float)
    p = p - p[0]  # wrist at the origin
    scale = np.linalg.norm(p[9]) or 1.0  # wrist -> middle-finger MCP
    p = p / scale
    if mirror:
        p[:, 0] = -p[:, 0]  # a left hand looks like a right one
    return p.ravel()


def features(hands):
    """84 floats: slot 0 = right hand, slot 1 = left hand (mirrored); zeros where a hand is missing."""
    slots = [None, None]
    for hand in sorted(hands, key=lambda h: h["handedness"] != "Right"):
        want = 0 if hand["handedness"] == "Right" else 1
        slot = want if slots[want] is None else 1 - want  # WiLoR sometimes labels both hands the same
        if slots[slot] is None:
            slots[slot] = hand_vector(hand["points"], hand["handedness"] == "Left")
    return np.concatenate([s if s is not None else np.zeros(42) for s in slots])


def add_sample(label, hands):
    os.makedirs(DIR, exist_ok=True)
    with open(SAMPLES, "a", encoding="utf-8") as f:
        f.write(json.dumps({"label": label, "hands": hands, "at": int(time.time() * 1000)}) + "\n")


def load_samples():
    if not os.path.exists(SAMPLES):
        return []
    with open(SAMPLES, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def counts():
    return dict(Counter(s["label"] for s in load_samples()))


def train():
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from confusion import report

    samples = load_samples()
    per_class = Counter(s["label"] for s in samples)
    if len(per_class) < 2:
        raise ValueError("record at least 2 signs first (e.g. one sign plus NOTHING)")
    X = np.array([features(s["hands"]) for s in samples])
    y = np.array([s["label"] for s in samples])
    small = min(per_class.values()) < 5

    def make():
        if small:
            return KNeighborsClassifier(n_neighbors=1)
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=0)

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    except ValueError:  # too few samples to stratify
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    held = make().fit(X_tr, y_tr)
    table = {label: Counter() for label in sorted(per_class)}
    for true, pred in zip(y_te, held.predict(X_te)):
        table[true][pred] += 1
    table = {k: v for k, v in table.items() if v}
    print(f"{len(samples)} samples, {len(per_class)} classes, {'KNN' if small else 'MLP'}; held-out 20%:")
    pairs = report(table)
    model = make().fit(X, y)  # ship the model trained on everything
    os.makedirs(DIR, exist_ok=True)
    with open(MODEL + ".tmp", "wb") as f:
        pickle.dump(model, f)
    os.replace(MODEL + ".tmp", MODEL)  # predict() never sees a half-written file
    tested = sum(sum(r.values()) for r in table.values())
    return {
        "samples": len(samples),
        "classifier": "KNN" if small else "MLP",
        "accuracy": round(sum(r[k] for k, r in table.items()) / tested, 3),
        "per_class": {k: round(r[k] / sum(r.values()), 3) for k, r in table.items()},
        "confused": [[a, b, n] for (a, b), n in pairs],
        "advice": "record more reps or swap one sign" if pairs else "",
    }


def ready():
    return os.path.exists(MODEL)


def model():
    mtime = os.path.getmtime(MODEL)
    if mtime != loaded["mtime"]:
        with open(MODEL, "rb") as f:
            loaded["model"] = pickle.load(f)
        loaded["mtime"] = mtime
        window.clear()  # classes may have changed
    return loaded["model"]


def predict(hands):
    """{"label", "confidence"} averaged over the last WINDOW frames, or None (no hands, unsure, NOTHING)."""
    if not hands or not ready():
        window.clear()
        return None
    m = model()
    window.append(m.predict_proba([features(hands)])[0])
    mean = np.mean(window, axis=0)
    best = int(np.argmax(mean))
    label = str(m.classes_[best])
    if mean[best] < MIN_CONFIDENCE or label == "NOTHING":
        return None
    return {"label": label, "confidence": round(float(mean[best]), 3)}


def selftest():
    import tempfile
    global DIR, SAMPLES, MODEL
    DIR = tempfile.mkdtemp()
    SAMPLES, MODEL = os.path.join(DIR, "samples.jsonl"), os.path.join(DIR, "model.pkl")
    rng = np.random.default_rng(0)
    base = {label: rng.uniform(0, 1, (21, 2)) for label in ("HELLO", "YES", "NOTHING")}
    for pts in base.values():
        pts[0], pts[9] = (0.5, 1.0), (0.5, 0.3)  # a real-sized palm keeps the scale stable

    def fake(label, handedness="Right", shift=0.0):
        pts = base[label] * 0.2 + shift + rng.normal(0, 0.003, (21, 2))
        if handedness == "Left":
            pts[:, 0] = 2 * shift - pts[:, 0]  # mirror image of the right hand
        return [{"handedness": handedness, "points": pts.tolist()}]

    # features are translation/scale/mirror invariant
    a = features(fake("HELLO", shift=0.1))
    assert np.allclose(a, features(fake("HELLO", shift=0.5)), atol=0.1)
    assert np.allclose(a[:42], features(fake("HELLO", "Left", shift=0.4))[42:], atol=0.1)  # left slot, mirrored
    assert features([]).shape == (84,) and not features([]).any()
    assert predict(fake("HELLO")) is None, "no model yet"
    for label in base:
        for i in range(30):
            add_sample(label, fake(label, "Left" if i % 3 == 0 else "Right", rng.uniform(0.1, 0.6)))
    assert counts() == {"HELLO": 30, "YES": 30, "NOTHING": 30}
    summary = train()
    assert summary["classifier"] == "MLP" and summary["accuracy"] >= 0.9, summary
    for label in ("HELLO", "YES"):
        for _ in range(WINDOW):
            got = predict(fake(label, shift=0.3))
        assert got and got["label"] == label and got["confidence"] >= MIN_CONFIDENCE, (label, got)
    for _ in range(WINDOW):
        got = predict(fake("NOTHING", shift=0.3))
    assert got is None, got
    assert predict([]) is None
    print("\nselftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif sys.argv[1:] == ["train"]:
        print(json.dumps(train(), indent=2))
    else:
        sys.exit(__doc__)
