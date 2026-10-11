"""Speaks a sentence with an ElevenLabs voice for the glasses (hub GET /tts?text=...).

Needs ELEVENLABS_API_KEY. ELEVENLABS_VOICE_ID picks the voice (default: Rachel).
Audio is cached on disk by text, so repeated phrases cost one API call and play instantly.
"""
import hashlib
import json
import os
import threading
import wave
from urllib.request import Request, urlopen

VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")  # lowest-latency model
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts")


def available():
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def path(text):
    return os.path.join(CACHE, hashlib.sha1(("%s|%s|en|%s" % (VOICE, MODEL, text)).encode()).hexdigest() + ".pcm")


def speak(text):
    """Raw 16 kHz mono 16-bit PCM of `text`: the glasses play it without a decoder (MP3 support is undocumented)."""
    cached = path(text)
    if os.path.exists(cached):
        with open(cached, "rb") as f:
            return f.read()
    req = Request("https://api.elevenlabs.io/v1/text-to-speech/%s?output_format=pcm_16000" % VOICE,
                  data=json.dumps({"text": text, "model_id": MODEL, "language_code": "en"}).encode(),
                  headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json",
                           "Accept": "audio/pcm"})
    with urlopen(req, timeout=15) as r:
        audio = r.read()
    os.makedirs(CACHE, exist_ok=True)
    with open(cached, "wb") as f:
        f.write(audio)
    return audio


def warm(lines):
    """Fetches the uncached `lines` in a daemon thread so their first /tts plays from disk. Returns the thread."""
    def run():
        fetched = 0
        for text in dict.fromkeys(lines):
            if os.path.exists(path(text)):
                continue
            try:
                speak(text)
                fetched += 1
            except Exception as e:
                print("voice cache: %r failed: %s" % (text, e))
        print("voice cache: %d lines ready (%d fetched)" % (sum(os.path.exists(path(t)) for t in set(lines)), fetched))
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


DEVICE = os.environ.get("TTS_DEVICE", "")  # substring of the output device to use (e.g. "Realtek"); empty = system default


def play(audio):
    """Also plays the PCM on the laptop's speaker: louder for people nearby than the glasses' open-ear speakers.
    With TTS_DEVICE set, plays on that output even when Windows' default is a monitor or headset."""
    if DEVICE:
        try:
            import numpy as np
            import sounddevice as sd
            index = next(i for i, d in enumerate(sd.query_devices())
                         if DEVICE.lower() in d["name"].lower() and d["max_output_channels"] > 0)
            sd.play(np.frombuffer(audio, np.int16), 16000, device=index)
            return True
        except Exception as error:  # device missing or sounddevice absent: fall through to the default output
            print("tts device", DEVICE, "unavailable:", error)
    try:
        import winsound
    except ImportError:  # not Windows: the glasses alone speak
        return False
    # winsound cannot play async from memory, so the WAV lives in the cache next to the PCM.
    path = os.path.join(CACHE, hashlib.sha1(audio).hexdigest() + ".wav")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio)
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    return path


if __name__ == "__main__":  # python tts.py: plays the newest cached line on the laptop
    import glob
    import time
    newest = max(glob.glob(os.path.join(CACHE, "*.pcm")), key=os.path.getmtime)
    with open(newest, "rb") as f:
        wav = play(f.read())
    assert wav and os.path.getsize(wav) == os.path.getsize(newest) + 44, wav
    time.sleep(3)
    print("played", newest)
