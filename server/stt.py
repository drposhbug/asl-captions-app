"""Captions from the laptop/chest mic with ElevenLabs speech-to-text (Scribe), for talkers the glasses' mic misses.

Each utterance is peak-normalised before upload, so a soft voice across a counter still transcribes.
Needs ELEVENLABS_API_KEY; without it, or with STT_BACKEND=local, or once ElevenLabs refuses the key (401/403),
utterances go to the offline faster-whisper model in stt_local.py instead.
Self-check (no API call, no model download): python stt.py --selftest
"""
import io
import json
import os
import sys
import uuid
import wave as wavfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import stt_local

RATE = 16000
MAX_GAIN = 30.0  # ~30 dB: lifts a whisper to full scale without turning silence into noise
MODEL = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1")
DISABLED = False  # set by the hub after a 401 (key without the speech_to_text permission)
LOCAL = os.environ.get("STT_BACKEND") == "local"  # also set for the rest of the process once ElevenLabs refuses


def available():
    return bool(os.environ.get("ELEVENLABS_API_KEY")) or stt_local.available()


def to_wav(wave):
    """Mono float samples -> 16-bit WAV bytes, peak-normalised so quiet talkers reach full scale."""
    import numpy as np
    peak = float(np.abs(wave).max()) if len(wave) else 0.0
    if peak > 0:
        wave = wave * min(0.9 / peak, MAX_GAIN)
    pcm = (np.clip(wave, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wavfile.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe(wave):
    """English text of the utterance, '' when nothing was said."""
    global LOCAL
    if LOCAL or not os.environ.get("ELEVENLABS_API_KEY"):
        return stt_local.transcribe(wave)
    try:
        return _elevenlabs(wave)
    except HTTPError as e:
        if e.code not in (401, 403) or not stt_local.available():
            raise
        LOCAL = True
        print("laptop captions: ElevenLabs refused, using local whisper")
        return stt_local.transcribe(wave)


def _elevenlabs(wave):
    boundary = uuid.uuid4().hex

    def part(name, value, filename=None, ctype=None):
        head = 'Content-Disposition: form-data; name="%s"' % name + ('; filename="%s"' % filename if filename else "")
        if ctype:
            head += "\r\nContent-Type: %s" % ctype
        data = value if isinstance(value, bytes) else value.encode()
        return ("--%s\r\n%s\r\n\r\n" % (boundary, head)).encode() + data + b"\r\n"

    body = (part("model_id", MODEL) + part("language_code", "eng") + part("tag_audio_events", "false")
            + part("file", to_wav(wave), "utterance.wav", "audio/wav") + ("--%s--\r\n" % boundary).encode())
    req = Request("https://api.elevenlabs.io/v1/speech-to-text", data=body,
                  headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"],
                           "Content-Type": "multipart/form-data; boundary=" + boundary})
    with urlopen(req, timeout=20) as r:
        return json.loads(r.read()).get("text", "").strip()


def selftest():
    import numpy as np
    t = np.arange(RATE) / RATE
    quiet = 0.01 * np.sin(2 * np.pi * 440 * t)
    with wavfile.open(io.BytesIO(to_wav(quiet))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, RATE, RATE)
        peak = np.abs(np.frombuffer(w.readframes(RATE), "<i2")).max() / 32767
    assert 0.28 < peak < 0.32, peak  # 0.01 * MAX_GAIN: a quiet signal is lifted, capped at 30x
    with wavfile.open(io.BytesIO(to_wav(quiet * 100))) as w:
        peak = np.abs(np.frombuffer(w.readframes(RATE), "<i2")).max() / 32767
    assert 0.88 < peak < 0.92, peak  # a loud signal is held just under full scale
    assert len(to_wav(np.zeros(100))) == 44 + 200  # silence stays silence

    # dispatch, without the network or the whisper model
    global LOCAL, _elevenlabs
    calls = []
    stt_local.transcribe = lambda w: calls.append("local") or "local text"
    stt_local.available = lambda: True
    _elevenlabs = lambda w: calls.append("cloud") or "cloud text"
    os.environ.pop("ELEVENLABS_API_KEY", None)
    LOCAL = False
    assert available() and transcribe(quiet) == "local text" and calls == ["local"]  # no key -> local
    os.environ["ELEVENLABS_API_KEY"] = "k"
    assert transcribe(quiet) == "cloud text" and calls[-1] == "cloud"  # key -> cloud

    def refuse(w):
        raise HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b""))
    _elevenlabs = refuse
    assert transcribe(quiet) == "local text" and LOCAL  # refused -> local, and stays local
    _elevenlabs = lambda w: "cloud text"
    assert transcribe(quiet) == "local text"
    LOCAL = False
    _elevenlabs = lambda w: (_ for _ in ()).throw(HTTPError("u", 500, "boom", {}, io.BytesIO(b"")))
    try:
        transcribe(quiet)
        assert False, "a 500 must still reach the hub"
    except HTTPError as e:
        assert e.code == 500 and not LOCAL
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
