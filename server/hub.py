"""Event hub for the glasses: captions/signs/sentences/alerts in, HUD mirror page and sound alerts out.

Run:        python hub.py [--sounds] [--yamnet]
            --sounds  listen on the laptop mic and push sound alerts (loudness detector by default)
            --yamnet  with --sounds, classify sounds with YAMNet (needs tensorflow, tensorflow_hub)
Self-check: python hub.py --selftest
"""
import json
import os
import re
import statistics
import struct
import sys
import threading
import time
import zlib
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PORT = int(os.environ.get("PORT", 8766))  # App Platform sets PORT
# Laptop hub with --forward <cloud hub url>: mic alarms are re-posted to the cloud hub the phone and glasses use.
FORWARD_URL = sys.argv[sys.argv.index("--forward") + 1].rstrip("/") if "--forward" in sys.argv else None
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "events.log")
HUD = os.path.join(HERE, "hud.html")
TYPES = {"caption", "sign", "sentence", "alert", "status"}
MAX_BODY = 64 * 1024
HAND_URL = "http://127.0.0.1:8765"  # hand server (or forward.py in front of it)
NAME_RE = re.compile(r"[A-Z0-9_-]{1,20}")  # enrolled voice names

lock = threading.RLock()  # re-entrant: push_alert holds it around add_event
events = deque(maxlen=500)
alerts = deque(maxlen=500)
# Start from the clock so ids keep rising across hub restarts and clients never wait on a stale `since`.
last_id = int(time.time() * 1000)


def now_ms():
    return int(time.time() * 1000)


def add_event(ev):
    global last_id
    with lock:
        last_id += 1
        ev = {"id": last_id, **ev}
        events.append(ev)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
    return ev


def push_alert(kind, label, confidence, direction=None):
    # alert and its mirror event share one id so the HUD can de-duplicate them
    with lock:
        ev = add_event({"type": "alert", "text": label, "confidence": round(confidence, 3), "label": kind, "at": now_ms()})
        alerts.append({"id": ev["id"], "kind": kind, "label": label, "confidence": ev["confidence"], "at": ev["at"],
                       "direction": direction})  # 'left' | 'center' | 'right' | None (mono mic)


# --- who is talking (speaker.py) -----------------------------------------------
mic = deque(maxlen=40)  # (end time, ~0.975 s window): recent mic audio shared by speaker id and enrollment
mic_on = False
NO_SPEAKER = {"name": None, "confidence": 0.0, "at": None, "side": None}
speaker = NO_SPEAKER
# Which side the current voice comes from (stereo mic only): {"side": 'left'|'center'|'right', "at": ms}
voice_side = {"side": None, "at": 0}
last_speaker_event = [None, 0.0]  # name, time.time()
HOLD_S = 4.0


def note_speaker(result):
    global speaker
    speaker = {"name": result["name"], "confidence": result["confidence"], "at": now_ms()}
    name = result["name"]
    if name and name != last_speaker_event[0] and time.time() - last_speaker_event[1] >= HOLD_S:
        last_speaker_event[:] = [name, time.time()]
        add_event({"type": "status", "text": "speaker: %s %.2f" % (name, result["confidence"]),
                   "confidence": result["confidence"], "label": name, "at": now_ms()})


def current_speaker():
    s = speaker
    fresh = s if s["at"] and now_ms() - s["at"] <= HOLD_S * 1000 else NO_SPEAKER
    side = voice_side["side"] if now_ms() - voice_side["at"] <= HOLD_S * 1000 else None
    return dict(fresh, side=side) if side else fresh


def voice_name(q):
    n = q.get("name", [""])[0].strip().upper()
    return n if NAME_RE.fullmatch(n) else None


def png(n):
    """Square PNG icon: green block on black, generated so the repo needs no binary assets."""
    m = n // 4
    black, green = b"\0\0\0" * n, b"\0\0\0" * m + b"\x40\xff\x5e" * (n - 2 * m) + b"\0\0\0" * m
    raw = b"".join(b"\0" + (green if m <= y < n - m else black) for y in range(n))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


ICONS = {"/icon-%d.png" % n: png(n) for n in (192, 512)}
MANIFEST = {"name": "ASL Alerts", "short_name": "ASL Alerts", "start_url": "/", "display": "standalone",
            "background_color": "#000000", "theme_color": "#000000",
            "icons": [{"src": p, "sizes": "%dx%d" % (n, n), "type": "image/png"} for p, n in zip(ICONS, (192, 512))]}
SW = b"self.addEventListener('fetch', () => {});\n"  # no-op; only here so Chrome offers install


def since(ring, query):
    try:
        n = int(query.get("since", ["0"])[0])
    except ValueError:
        n = 0
    with lock:
        items = [e for e in ring if e["id"] > n]
    return items, (items[-1]["id"] if items else n)


def parse_event(body):
    ev = json.loads(body)
    if not isinstance(ev, dict) or ev.get("type") not in TYPES or not isinstance(ev.get("text"), str):
        raise ValueError("need type in %s and text string" % sorted(TYPES))
    conf, label, at = ev.get("confidence"), ev.get("label"), ev.get("at")
    if conf is not None and not isinstance(conf, (int, float)):
        raise ValueError("confidence must be a number or null")
    if label is not None and not isinstance(label, str):
        raise ValueError("label must be a string or null")
    if at is not None and not isinstance(at, (int, float)):
        raise ValueError("at must be ms epoch or null")
    return {"type": ev["type"], "text": ev["text"][:2000], "confidence": conf, "label": label, "at": at or now_ms()}


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def relay(self, method):
        """Sign recording/training lives on the hand server; relay so the dashboard needs one origin."""
        try:
            with urlopen(Request(HAND_URL + self.path, data=b"" if method == "POST" else None, method=method), timeout=60) as r:
                return self.send(200, r.read())
        except HTTPError as e:
            return self.send(e.code, e.read())
        except Exception as e:
            return self.send(502, {"error": "hand server unreachable: %s" % e})

    def do_OPTIONS(self):
        self.send(204, b"", "text/plain")

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            with open(HUD, "rb") as f:
                return self.send(200, f.read(), "text/html; charset=utf-8")
        if url.path == "/manifest.json":
            return self.send(200, MANIFEST, "application/manifest+json")
        if url.path == "/sw.js":
            return self.send(200, SW, "text/javascript")
        if url.path in ICONS:
            return self.send(200, ICONS[url.path], "image/png")
        if url.path == "/events":
            items, nxt = since(events, q)
            return self.send(200, {"events": items, "next": nxt})
        if url.path == "/alerts":
            items, nxt = since(alerts, q)
            return self.send(200, {"alerts": items, "next": nxt})
        if url.path == "/preview.jpg":
            # Relay the hand server's camera preview so one origin (and the tunnel) serves the whole dashboard.
            try:
                with urlopen(HAND_URL + "/preview.jpg", timeout=2) as r:
                    return self.send(200, r.read(), "image/jpeg")
            except Exception:
                return self.send(404, {"error": "no camera frame yet"})
        if url.path == "/record":
            return self.relay("GET")
        if url.path == "/speaker":
            return self.send(200, current_speaker())
        if url.path == "/speaker/profiles":
            import speaker as voices
            return self.send(200, voices.profiles())
        if url.path == "/tts":  # ElevenLabs voice for a sentence the glasses speak
            text = q.get("text", [""])[0].strip()
            if not 0 < len(text) <= 300:
                return self.send(400, {"error": "text must be 1..300 chars"})
            import tts
            if not tts.available():
                return self.send(503, {"error": "set ELEVENLABS_API_KEY"})
            try:
                return self.send(200, tts.speak(text), "audio/mpeg")
            except Exception as e:  # any API failure: the glasses fall back to their built-in voice
                print("tts failed:", e)
                return self.send(502, {"error": str(e)[:200]})
        if url.path == "/flight":
            try:
                import flight
            except Exception as e:
                print("flight import failed:", e)
                return self.send(503, {"error": "flight module unavailable"})
            try:
                return self.send(200, flight.status(q.get("number", [""])[0].strip().upper()))
            except Exception as e:
                return self.send(502, {"error": str(e)})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path in ("/speaker/enroll", "/speaker/clear"):
            import speaker as voices
            q = parse_qs(url.query)
            name = voice_name(q)
            if not name:
                return self.send(400, {"error": "name must be [A-Z0-9_-]{1,20}"})
            if url.path == "/speaker/clear":
                voices.clear(name)
                return self.send(200, {"ok": True})
            try:
                seconds = int(q.get("seconds", ["5"])[0])
                if not 1 <= seconds <= 30:
                    raise ValueError
            except ValueError:
                return self.send(400, {"error": "seconds must be 1..30"})
            if not mic_on:
                return self.send(503, {"error": "mic off: run hub.py --sounds"})
            t0 = time.time()
            time.sleep(seconds + WINDOW / RATE + 0.2)  # let the last window land
            waves = [w for t, w in list(mic) if t - WINDOW / RATE / 2 >= t0][: int(seconds * RATE / WINDOW) + 1]  # windows mostly after the request
            if not waves:
                return self.send(503, {"error": "no mic audio arrived"})
            import numpy as np
            try:
                return self.send(200, {"ok": True, "samples": voices.enroll(name, np.concatenate(waves))})
            except Exception as e:  # e.g. resemblyzer/torch missing
                return self.send(500, {"error": str(e)[:200]})
        if urlparse(self.path).path in ("/record", "/train"):
            return self.relay("POST")
        if urlparse(self.path).path == "/test-alert":
            push_alert("fire_alarm", "FIRE ALARM (TEST)", 1.0)
            return self.send(200, {"ok": True})
        if urlparse(self.path).path == "/alerts":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                if not 0 < length <= MAX_BODY:
                    raise ValueError("body must be 1..%d bytes" % MAX_BODY)
                a = json.loads(self.rfile.read(length))
                push_alert(str(a["kind"])[:40], str(a["label"])[:80], float(a["confidence"]))
            except (ValueError, KeyError, TypeError) as e:
                return self.send(400, {"error": "alert needs kind, label, confidence: %s" % e})
            return self.send(200, {"ok": True})
        if urlparse(self.path).path == "/say":  # demo fallback: glasses speak `text` (kind "say" alert)
            length = int(self.headers.get("Content-Length") or 0)
            try:
                if not 0 < length <= MAX_BODY:
                    raise ValueError("body must be 1..%d bytes" % MAX_BODY)
                a = json.loads(self.rfile.read(length))
                text, glosses = str(a["text"])[:200], str(a["glosses"])[:200]
            except (ValueError, KeyError, TypeError) as e:
                return self.send(400, {"error": "say needs text, glosses: %s" % e})
            push_alert("say", text, 1.0)
            add_event({"type": "sign", "text": glosses, "confidence": 1.0, "label": None, "at": now_ms()})  # after, so the HUD keeps it
            return self.send(200, {"ok": True})
        if urlparse(self.path).path == "/polish":  # Claude cleans one finished caption utterance
            length = int(self.headers.get("Content-Length") or 0)
            try:
                if not 0 < length <= MAX_BODY:
                    raise ValueError("body must be 1..%d bytes" % MAX_BODY)
                a = json.loads(self.rfile.read(length))
                text = str(a["text"])[:500]
                context = [str(c)[:300] for c in a.get("context", [])][-3:]
                location = str(a.get("location") or "")[:80]
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                return self.send(400, {"error": "polish needs text: %s" % e})
            import polish
            if not polish.available():
                return self.send(503, {"error": "set ANTHROPIC_API_KEY and pip install anthropic"})
            try:
                cleaned = polish.clean(text, context, location)
            except Exception as e:  # any API failure: the glasses keep the raw caption
                print("polish failed:", e)
                return self.send(502, {"error": str(e)[:200]})
            add_event({"type": "caption", "text": cleaned, "confidence": None, "label": "CLEANED", "at": now_ms()})
            return self.send(200, {"text": cleaned})
        if urlparse(self.path).path != "/events":
            return self.send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            return self.send(413, {"error": "body must be 1..%d bytes" % MAX_BODY})
        try:
            ev = parse_event(self.rfile.read(length))
        except ValueError as e:  # JSONDecodeError is a ValueError
            return self.send(400, {"error": str(e)})
        add_event(ev)
        self.send(200, {"ok": True})

    def log_message(self, *args):
        pass


# --- sound alerts -------------------------------------------------------------
RATE = 16000
WINDOW = int(0.975 * RATE)
DEBOUNCE_S = 3.0
MIC_SPACING_M = float(os.environ.get("MIC_SPACING_M", "0.08"))  # distance between the two mics for --stereo
YAMNET_KINDS = {
    "Fire alarm": "fire_alarm", "Smoke detector, smoke alarm": "fire_alarm",
    "Siren": "siren", "Civil defense siren": "siren", "Police car (siren)": "siren",
    "Ambulance (siren)": "siren", "Fire engine, fire truck (siren)": "siren",
    "Vehicle horn, car horn, honking": "car_horn", "Car alarm": "car_horn",
    "Knock": "door_knock", "Doorbell": "door_knock", "Ding-dong": "door_knock",
    # Other sounds a Deaf wearer needs to know about right away.
    "Shatter": "glass_break", "Glass": "glass_break",
    "Screaming": "scream", "Shout": "scream", "Yell": "scream", "Children shouting": "scream",
    "Gunshot, gunfire": "gunshot", "Explosion": "gunshot", "Fireworks": "gunshot",
    "Baby cry, infant cry": "baby_cry", "Crying, sobbing": "baby_cry",
    "Dog": "dog", "Bark": "dog",
    "Alarm": "alarm", "Alarm clock": "alarm", "Buzzer": "alarm", "Emergency vehicle": "siren",
}
LABELS = {
    "fire_alarm": "FIRE ALARM", "siren": "SIREN", "car_horn": "CAR HORN", "door_knock": "DOOR KNOCK",
    "glass_break": "GLASS BREAKING", "scream": "SOMEONE SCREAMING", "gunshot": "LOUD BANG",
    "baby_cry": "BABY CRYING", "dog": "DOG BARKING", "alarm": "ALARM",
}


def yamnet_classifier():
    import csv
    import tensorflow as tf
    import tensorflow_hub as hub
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    with tf.io.gfile.GFile(model.class_map_path().numpy()) as f:
        names = [row["display_name"] for row in csv.DictReader(f)]
    idx = [(i, YAMNET_KINDS[n]) for i, n in enumerate(names) if n in YAMNET_KINDS]

    speech_i = names.index("Speech")

    def classify(wave):
        scores = model(wave)[0].numpy().max(axis=0)  # max over YAMNet's ~0.48 s frames
        classify.speech = float(scores[speech_i])
        best = {}
        for i, kind in idx:
            best[kind] = max(best.get(kind, 0.0), float(scores[i]))
        return [(k, LABELS[k], s) for k, s in best.items() if s >= 0.3]
    return classify


def loudness_classifier():
    import numpy as np
    history = deque(maxlen=30)  # ~30 s of window RMS values

    def classify(wave):
        rms = float(np.sqrt(np.mean(wave ** 2)))
        hits = []
        # shortcut: loudness spike only, no sound type; use --yamnet for real classes
        median = statistics.median(history) if len(history) >= 5 else None
        # shortcut: "speech" = steady sound above the room's median; upgrade with --yamnet
        classify.speech = 1.0 if median is not None and rms > max(0.01, 2 * median) else 0.0
        if median is not None and rms > 4 * median and rms > 0.01:
            hits.append(("other", "LOUD SOUND", min(1.0, rms / (8 * median))))  # 4x -> 0.5, 8x+ -> 1.0
        history.append(rms)
        return hits
    return classify


def identify_speaker():
    import numpy as np
    import speaker as voices
    try:
        note_speaker(voices.identify(np.concatenate([w for _, w in list(mic)[-3:]])[-2 * RATE:]))
    except Exception as e:  # speaker id is a bonus; never stop the sound alerts
        print("speaker id failed:", e)


def listen(use_yamnet):
    global mic_on
    import sounddevice as sd
    classify = yamnet_classifier() if use_yamnet else loudness_classifier()
    print("sound alerts on (%s)" % ("YAMNet" if use_yamnet else "loudness detector"))
    last = {}
    # --stereo: read both channels of a stereo mic (chest camera, laptop array) to tell which side a sound is on.
    stereo_mode = "--stereo" in sys.argv
    with sd.InputStream(samplerate=RATE, channels=2 if stereo_mode else 1, dtype="float32") as stream:
        while True:
            mic_on = True
            frames = stream.read(WINDOW)[0]
            wave = frames.mean(axis=1) if stereo_mode else frames[:, 0]
            mic.append((time.time(), wave))
            hits = classify(wave)
            if getattr(classify, "speech", 0) >= 0.5:
                if stereo_mode:  # in a group, which side the talker is on matters as much as who it is
                    import direction
                    voice_side.update(side=direction.direction(frames, RATE, MIC_SPACING_M)[0], at=now_ms())
                identify_speaker()
            for kind, label, conf in hits:
                t = time.time()
                if t - last.get(kind, 0) >= DEBOUNCE_S:
                    last[kind] = t
                    side = None
                    if stereo_mode:
                        import direction
                        side = direction.direction(frames, RATE, MIC_SPACING_M)[0]
                    push_alert(kind, label, conf, side)
                    print("ALERT", kind, round(conf, 2))
                    if FORWARD_URL:
                        try:
                            body = json.dumps({"kind": kind, "label": label, "confidence": conf}).encode()
                            urlopen(Request(FORWARD_URL + "/alerts", data=body, headers={"Content-Type": "application/json"}), timeout=5)
                        except Exception as e:  # the local HUD already has the alert; never stop listening
                            print("forward failed:", e)


def selftest():
    import tempfile
    global LOG
    LOG = os.path.join(tempfile.mkdtemp(), "events.log")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % server.server_address[1]

    def call(path, body=None):
        req = Request(base + path, data=body and json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req) as r:
            assert r.headers["Access-Control-Allow-Origin"] == "*"
            return r.read()

    assert json.loads(call("/events", {"type": "sentence", "text": "hello there", "confidence": 0.93, "label": None, "at": None})) == {"ok": True}
    got = json.loads(call("/events?since=0"))
    assert got["events"][0]["text"] == "hello there" and got["next"] == got["events"][0]["id"], got
    assert json.loads(call("/events?since=%d" % got["next"]))["events"] == []
    push_alert("fire_alarm", "FIRE ALARM", 0.8)
    a = json.loads(call("/alerts?since=0"))
    assert a["alerts"][0]["kind"] == "fire_alarm" and a["next"] == a["alerts"][0]["id"], a
    assert b"<title>" in call("/")
    assert json.loads(call("/manifest.json"))["display"] == "standalone"
    assert b"addEventListener" in call("/sw.js") and call("/icon-512.png")[1:4] == b"PNG"
    with urlopen(Request(base + "/test-alert", data=b"", method="POST")) as r:
        assert json.loads(r.read()) == {"ok": True}
    t = json.loads(call("/alerts?since=%d" % a["next"]))["alerts"]
    assert [x["label"] for x in t] == ["FIRE ALARM (TEST)"] and t[0]["confidence"] == 1.0, t
    assert json.loads(call("/events?since=0"))["events"][-1]["id"] == t[0]["id"]  # shared id for HUD de-dup
    try:
        call("/events", {"type": "nope", "text": "x"})
        raise AssertionError("bad event accepted")
    except Exception as e:
        assert getattr(e, "code", None) == 400, e
    assert json.loads(call("/say", {"text": "Hello!", "glosses": "HELLO"})) == {"ok": True}
    s = json.loads(call("/alerts?since=%d" % t[0]["id"]))["alerts"]
    assert [(x["kind"], x["label"]) for x in s] == [("say", "Hello!")], s
    assert [(e["type"], e["text"]) for e in json.loads(call("/events?since=%d" % t[0]["id"]))["events"]] == [("alert", "Hello!"), ("sign", "HELLO")]
    try:
        call("/say", {"text": "x"})
        raise AssertionError("say without glosses accepted")
    except Exception as e:
        assert getattr(e, "code", None) == 400, e
    with open(LOG) as f:
        assert len(f.read().splitlines()) == 5
    global HAND_URL
    hand = ThreadingHTTPServer(("127.0.0.1", 0), type("Stub", (Handler,), {  # fake hand server echoing the path
        "do_GET": lambda h: h.send(200, {"path": h.path}), "do_POST": lambda h: h.send(400, {"path": h.path})}))
    threading.Thread(target=hand.serve_forever, daemon=True).start()
    HAND_URL = "http://127.0.0.1:%d" % hand.server_address[1]
    assert json.loads(call("/record?x=1")) == {"path": "/record?x=1"}
    try:
        call("/train", {})
        raise AssertionError("relay hid the hand server's error")
    except HTTPError as e:
        assert e.code == 400 and json.loads(e.read()) == {"path": "/train"}, e
    hand.shutdown()
    # speaker id: stub identify and enroll so the test needs no torch or mic
    import numpy as np
    import speaker as voices
    voices.DIR = tempfile.mkdtemp()
    assert json.loads(call("/speaker")) == NO_SPEAKER
    voices.identify = lambda w: {"name": "ALICE", "confidence": 0.83}
    mic.extend((time.time(), np.zeros(WINDOW, np.float32)) for _ in range(3))
    identify_speaker()
    sp = json.loads(call("/speaker"))
    assert sp["name"] == "ALICE" and sp["confidence"] == 0.83 and sp["at"], sp
    assert json.loads(call("/events?since=0"))["events"][-1]["text"] == "speaker: ALICE 0.83"
    np.save(os.path.join(voices.DIR, "ALICE.npy"), np.zeros((2, 256), np.float32))
    assert json.loads(call("/speaker/profiles")) == {"ALICE": 2}
    for bad in ("/speaker/clear?name=bad%20name", "/speaker/clear?name=", "/speaker/enroll?name=" + "X" * 21):
        try:
            call(bad, {})
            raise AssertionError("bad name accepted: " + bad)
        except HTTPError as e:
            assert e.code == 400, (bad, e)
    try:
        call("/speaker/enroll?name=bob&seconds=1", {})
        raise AssertionError("enroll without mic accepted")
    except HTTPError as e:
        assert e.code == 503, e
    global mic_on
    mic_on = True
    voices.enroll = lambda name, wave: (name, len(wave))
    feeder = threading.Timer(1.5, lambda: mic.append((time.time(), np.zeros(WINDOW, np.float32))))
    feeder.start()
    r = json.loads(call("/speaker/enroll?name=bob&seconds=1", {}))
    assert r == {"ok": True, "samples": ["BOB", WINDOW]}, r
    assert json.loads(call("/speaker/clear?name=alice", {})) == {"ok": True}
    assert json.loads(call("/speaker/profiles")) == {}
    print("selftest ok")
    server.shutdown()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    if "--sounds" in sys.argv:
        threading.Thread(target=listen, args=("--yamnet" in sys.argv,), daemon=True).start()
    print("hub on http://0.0.0.0:%d" % PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
