"""Speaks a sentence with an ElevenLabs voice for the glasses (hub GET /tts?text=...).

Needs ELEVENLABS_API_KEY. ELEVENLABS_VOICE_ID picks the voice (default: Rachel).
Audio is cached on disk by text, so repeated phrases cost one API call and play instantly.
"""
import hashlib
import json
import os
from urllib.request import Request, urlopen

VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")  # lowest-latency model
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts")


def available():
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def speak(text):
    """Raw 16 kHz mono 16-bit PCM of `text`: the glasses play it without a decoder (MP3 support is undocumented)."""
    path = os.path.join(CACHE, hashlib.sha1(("%s|%s|%s" % (VOICE, MODEL, text)).encode()).hexdigest() + ".pcm")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    req = Request("https://api.elevenlabs.io/v1/text-to-speech/%s?output_format=pcm_16000" % VOICE,
                  data=json.dumps({"text": text, "model_id": MODEL}).encode(),
                  headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json",
                           "Accept": "audio/pcm"})
    with urlopen(req, timeout=15) as r:
        audio = r.read()
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as f:
        f.write(audio)
    return audio
