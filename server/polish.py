"""Cleans one finished caption utterance with Claude: fixes misheard words, grammar and duplicates.

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile). CAPTION_MODEL picks the model.
Self-check (no API call): python polish.py --selftest
"""
import os
import sys

# Sonnet 5.5: fastest and as accurate as Opus on caption cleanup; Haiku leaked its reasoning into the caption.
MODEL = os.environ.get("CAPTION_MODEL", "claude-sonnet-5-5")
# Server-side refusal fallback only exists for these models; Haiku has none.
FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}

SYSTEM = (
    "You clean up live speech-recognition captions shown to a Deaf reader on AR glasses. "
    "Fix misheard words, grammar and punctuation so the line reads naturally. "
    "The recognizer often emits a word twice or as two similar-sounding guesses "
    "(e.g. 'phrase verse' when only 'phrase' was said): keep the one that fits and drop the other. "
    "Keep the speaker's meaning and wording; never add new content or answer the speaker. "
    "Earlier lines are context only. If an image is attached it is what the wearer is looking at right now: "
    "use it to resolve names, numbers, places and objects. A location, when given, says roughly where the wearer is "
    "(e.g. an airport or a cafe); use it the same way. Reply with only the cleaned caption, nothing else."
)
FRAME_URL = "http://127.0.0.1:8765/frame.jpg"  # the glasses' latest camera frame, via the hand server
FRAME_MAX_AGE_S = 15


def latest_frame():
    """Base64 JPEG of what the glasses see now, or None if the camera is off or the frame is stale."""
    import base64
    from urllib.request import urlopen

    try:
        with urlopen(FRAME_URL, timeout=1) as r:
            if float(r.headers.get("X-Frame-Age", "1e9")) > FRAME_MAX_AGE_S:
                return None
            return base64.b64encode(r.read()).decode()
    except Exception:
        return None

_client = None


def available():
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def build_prompt(text, context, location=""):
    lines = [f"Wearer location (lat, lon): {location}"] if location else []
    lines += [f"Earlier: {c}" for c in context[-3:] if c]
    lines.append(f"Caption: {text}")
    return "\n".join(lines)


def content(text, context, frame_b64, location=""):
    """User content: the scene image first (when fresh), then the caption and its context."""
    blocks = []
    if frame_b64:
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64}})
    blocks.append({"type": "text", "text": build_prompt(text, context, location)})
    return blocks


def clean(text, context=(), location=""):
    """Returns the cleaned caption, or the original text when Claude declines or returns nothing."""
    global _client
    import anthropic

    if _client is None:
        _client = anthropic.Anthropic(timeout=15.0, max_retries=1)
    kwargs = dict(
        model=MODEL,
        max_tokens=2048,  # room for adaptive thinking; the caption itself is a line
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content(text, list(context), latest_frame(), location)}],
    )
    if MODEL in FALLBACK_MODELS:
        response = _client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        response = _client.messages.create(**kwargs)
    if response.stop_reason == "refusal":
        return text
    out = "".join(block.text for block in response.content if block.type == "text").strip()
    return out or text


GLOSS_SYSTEM = (
    "You turn American Sign Language glosses into natural spoken English for a Deaf signer talking to a hearing person. "
    "ASL grammar differs from English: topic-comment order, no articles, no 'to be', time first, no tense endings. "
    "Fill in missing words, tense, articles and pronouns. Use the conversation context to pick the right meaning. "
    "Never add ideas the signer did not sign. Reply with one short first-person sentence and nothing else."
)


def sentence_from_glosses(glosses, context=()):
    """English sentence for a run of recognised glosses; the raw glosses if Claude declines."""
    global _client
    import anthropic

    if _client is None:
        _client = anthropic.Anthropic(timeout=15.0, max_retries=1)
    prompt = "\n".join([f"Heard: {c}" for c in list(context)[-3:] if c] + [f"Glosses: {' '.join(glosses)}"])
    response = _client.messages.create(
        model=MODEL, max_tokens=1024, system=GLOSS_SYSTEM, output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    )
    out = "".join(b.text for b in response.content if b.type == "text").strip()
    return out or " ".join(glosses)


def selftest():
    assert build_prompt("phrase verse", ["Hi there.", "", "How are you?"]) == (
        "Earlier: Hi there.\nEarlier: How are you?\nCaption: phrase verse"
    )
    assert build_prompt("ok", []) == "Caption: ok"
    assert build_prompt("ok", [], "1.0, 2.0") == "Wearer location (lat, lon): 1.0, 2.0\nCaption: ok"
    assert [b["type"] for b in content("ok", [], None)] == ["text"]
    assert [b["type"] for b in content("ok", [], "AAAA")] == ["image", "text"]
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(clean(" ".join(sys.argv[1:]) or "to grow type the gate is is over there"))
