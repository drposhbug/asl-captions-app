"""Cleans one finished caption utterance with Claude: fixes misheard words, grammar and duplicates.

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile). CAPTION_MODEL picks the model.
Self-check (no API call): python polish.py --selftest
"""
import os
import sys

MODEL = os.environ.get("CAPTION_MODEL", "claude-opus-5-5")
# Server-side refusal fallback only exists for these models; Haiku has none.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}

SYSTEM = (
    "You clean up live speech-recognition captions shown to a Deaf reader on AR glasses. "
    "Fix misheard words, grammar and punctuation so the line reads naturally. "
    "The recognizer often emits a word twice or as two similar-sounding guesses "
    "(e.g. 'phrase verse' when only 'phrase' was said): keep the one that fits and drop the other. "
    "Keep the speaker's meaning and wording; never add new content or answer the speaker. "
    "Earlier lines are context only. Reply with only the cleaned caption, nothing else."
)

_client = None


def available():
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def build_prompt(text, context):
    lines = [f"Earlier: {c}" for c in context[-3:] if c]
    lines.append(f"Caption: {text}")
    return "\n".join(lines)


def clean(text, context=()):
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
        messages=[{"role": "user", "content": build_prompt(text, list(context))}],
    )
    if MODEL in FALLBACK_MODELS:
        response = _client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        response = _client.messages.create(**kwargs)
    if response.stop_reason == "refusal":
        return text
    out = "".join(block.text for block in response.content if block.type == "text").strip()
    return out or text


def selftest():
    assert build_prompt("phrase verse", ["Hi there.", "", "How are you?"]) == (
        "Earlier: Hi there.\nEarlier: How are you?\nCaption: phrase verse"
    )
    assert build_prompt("ok", []) == "Caption: ok"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(clean(" ".join(sys.argv[1:]) or "to grow type the gate is is over there"))
