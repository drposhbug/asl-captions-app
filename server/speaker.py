"""Who is talking: resemblyzer voice embeddings matched against enrolled profiles in speakers/<NAME>.npy.

Self-check: python speaker.py --selftest
"""
import os
import sys

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers")
THRESHOLD = 0.75  # cosine similarity to a profile centroid; below it the speaker is unknown
RATE = 16000
_encoder = None


def embed(wave16k):
    global _encoder
    from resemblyzer import VoiceEncoder, preprocess_wav
    if _encoder is None:
        _encoder = VoiceEncoder("cpu", verbose=False)
    wave = np.asarray(wave16k, dtype=np.float32)
    trimmed = preprocess_wav(wave, source_sr=RATE)  # normalise volume, drop silence
    return _encoder.embed_utterance(trimmed if len(trimmed) >= RATE // 2 else wave)


def _path(name):
    return os.path.join(DIR, name + ".npy")


def enroll(name, wave):
    e = embed(wave)
    old = np.load(_path(name)) if os.path.exists(_path(name)) else np.empty((0, len(e)), np.float32)
    os.makedirs(DIR, exist_ok=True)
    np.save(_path(name), np.vstack([old, e]))
    return len(old) + 1


def profiles():
    if not os.path.isdir(DIR):
        return {}
    return {f[:-4]: len(np.load(os.path.join(DIR, f))) for f in sorted(os.listdir(DIR)) if f.endswith(".npy")}


def clear(name):
    if os.path.exists(_path(name)):
        os.remove(_path(name))


def identify(wave):
    names = list(profiles())
    if not names:
        return {"name": None, "confidence": 0.0}
    e = embed(wave)
    best, score = None, -1.0
    for n in names:
        c = np.load(_path(n)).mean(axis=0)
        s = float(np.dot(e, c) / (np.linalg.norm(e) * np.linalg.norm(c) + 1e-9))
        if s > score:
            best, score = n, s
    return {"name": best if score >= THRESHOLD else None, "confidence": round(score, 3)}


def selftest():
    import tempfile
    global DIR
    DIR = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    t = np.arange(2 * RATE) / RATE

    def voice(f0, seed):  # harmonic stack with vibrato + breath noise; crude but distinct per pitch
        r = np.random.default_rng(seed)
        ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))) / RATE
        w = sum(np.sin(k * ph) / k for k in range(1, 12)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t) ** 2)
        return (0.2 * w + 0.02 * r.standard_normal(len(t))).astype(np.float32)

    for seed in range(3):
        enroll("LOW", voice(95, seed))
        enroll("HIGH", voice(260, seed + 10))
    assert profiles() == {"LOW": 3, "HIGH": 3}, profiles()
    a, b = identify(voice(95, 99)), identify(voice(260, 98))
    n = identify((0.1 * rng.standard_normal(len(t))).astype(np.float32))
    print("low", a, "high", b, "noise", n)
    assert a["name"] == "LOW" and b["name"] == "HIGH", (a, b)
    assert n["name"] is None, n
    clear("LOW")
    assert profiles() == {"HIGH": 3}
    print("speaker selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
