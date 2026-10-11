"""Offline laptop captions with faster-whisper (CTranslate2, CPU int8): used by stt.py when the ElevenLabs key is
missing, refused, or STT_BACKEND=local. Same contract as stt.transcribe: 16 kHz mono float samples -> English text.

Install: python -m pip install faster-whisper   (base.en, ~75 MB, downloads to the Hugging Face cache on first use)
Env: STT_LOCAL_MODEL=tiny.en for ~2x faster, less accurate captions.
Bench (loads the model): python stt_local.py --bench [utterance.wav]
"""
import os
import sys
import threading
import time

RATE = 16000
MODELS = (os.environ.get("STT_LOCAL_MODEL", "base.en"), "tiny.en")  # first that loads wins
_model = None
_loading = threading.Lock()  # the hub transcribes each utterance on its own thread


def available():
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    global _model
    with _loading:
        if _model is None:
            from faster_whisper import WhisperModel
            t0 = time.time()
            for name in MODELS:
                try:
                    _model = WhisperModel(name, device="cpu", compute_type="int8")
                    break
                except Exception as e:
                    print("stt_local: %s unavailable: %s" % (name, e))
                    err = e
            else:
                raise err
            print("stt_local: %s loaded in %.1f s" % (name, time.time() - t0))
    return _model


def transcribe(wave):
    """English text of the utterance, '' when nothing was said."""
    import numpy as np
    wave = np.asarray(wave, np.float32)
    peak = float(np.abs(wave).max()) if len(wave) else 0.0
    if peak > 0:  # same lift as stt.to_wav so a soft voice clears whisper's VAD
        wave = wave * min(0.9 / peak, 30.0)
    segments, _ = _load().transcribe(wave, language="en", beam_size=1, vad_filter=True,
                                     condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segments).strip()


if __name__ == "__main__":
    if "--bench" in sys.argv:
        import numpy as np
        import wave as wavfile
        paths = [a for a in sys.argv[1:] if a.endswith(".wav")]
        clips = []
        for p in paths:
            with wavfile.open(p) as w:
                assert (w.getnchannels(), w.getframerate()) == (1, RATE), p
                clips.append((p, np.frombuffer(w.readframes(w.getnframes()), "<i2") / 32768.0))
        if not clips:
            clips.append(("3 s silence", np.zeros(3 * RATE, np.float32)))
        t0 = time.time()
        _load()
        for name, clip in clips:
            t1 = time.time()
            text = transcribe(clip)
            print("%-60s %.2fs audio -> %.2fs  %r" % (os.path.basename(name), len(clip) / RATE, time.time() - t1, text))
        print("total %.1f s" % (time.time() - t0))
