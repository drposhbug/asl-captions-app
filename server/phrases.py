"""The signer's own phrases, spoken word for word without the language model.

A Python port of PHRASES / matchPhrase in pages/index/signing.js so the laptop-camera mode of
hands_server.py speaks the same sentences as the glasses. Keep both tables identical.
Self-check (parses signing.js and compares): python phrases.py --selftest
"""
import os
import re
import sys

PHRASES = [
    ("WHICH GATE THIS", "Which gate did they say?"),
    ("ME DEAF", "I'm deaf."),
    ("WHERE GATE C4", "Where is gate C4?"),
    ("THIS FLIGHT DELAYED", "Is this flight delayed?"),
    ("WHEN FOOD ARRIVE", "When does the food arrive?"),
    ("ME LOST", "I'm lost."),
]


def key(glosses):
    """Sorted uppercase glosses, I -> ME, fingerspelled "C 4" -> C4 (ASL order varies)."""
    text = re.sub(r"\bC 4\b", "C4", re.sub(r"\bI\b", "ME", " ".join(glosses).upper()))
    return " ".join(sorted(re.split(r"\s+", text)))


PHRASE_BY_KEY = {key([glosses]): sentence for glosses, sentence in PHRASES}


def match(glosses):
    """The phrase sentence for these glosses, or None when they should go to the language model."""
    return PHRASE_BY_KEY.get(key(glosses))


def selftest():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pages", "index", "signing.js")
    with open(path, encoding="utf-8") as file:
        table = re.search(r"const PHRASES = \[\n(.*?)\n\];", file.read(), re.S).group(1)
    js = [(g, s1 or s2) for g, s1, s2 in re.findall(r"\['([^']*)', (?:'([^']*)'|\"([^\"]*)\")\]", table)]
    assert js == PHRASES, f"signing.js PHRASES differ:\n{js}\n{PHRASES}"
    # Same cases as pages/index/signing.test.mjs.
    assert match(["WHERE", "GATE", "C4"]) == "Where is gate C4?"
    assert match(["GATE", "C", "4", "WHERE"]) == "Where is gate C4?", "any order, fingerspelled C 4"
    assert match(["I", "DEAF"]) == "I'm deaf.", "I counts as ME"
    assert match(["ME", "WANT", "COFFEE"]) is None, "other signing goes to the language model"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
