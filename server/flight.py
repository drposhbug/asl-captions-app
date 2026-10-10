"""Flight status for the airport demo: one-line HUD messages for gate changes, delays and boarding.

status("UA123") -> {"flight", "status", "gate", "delay_minutes", "message"}.
With AVIATIONSTACK_KEY set it asks AviationStack (free tier, http only); otherwise a built-in
demo schedule advances every 60 s so the HUD shows changes without internet.
Usage: python flight.py UA123 | --selftest
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

KEY = os.environ.get("AVIATIONSTACK_KEY")
STEP_SECONDS = 60
# flight -> (gate, changed gate, delay minutes); the phase offset staggers flights so they differ at any moment
DEMO = {"UA123": ("B7", "B12", 25), "DL456": ("C3", "C9", 40), "AA789": ("A5", "A2", 15),
        "WN202": ("D1", "D4", 30), "B6303": ("E8", "E2", 50)}
# shortcut: real API has no "boarding" state; the demo invents it so the HUD has something to show
PHASES = ["scheduled", "boarding", "gate_change", "delayed", "departed"]


def _message(flight, status, gate, delay, gate_changed=False):
    parts = [flight]
    if gate:
        parts.append(f"GATE CHANGED TO {gate}" if gate_changed else f"GATE {gate}")
    if delay > 0:
        parts.append(f"DELAYED {delay} MIN")
    if status not in ("scheduled", "delayed") or delay <= 0:
        parts.append(status.upper())
    return " · ".join(parts)


def _result(flight, status, gate, delay, message):
    return {"flight": flight, "status": status, "gate": gate, "delay_minutes": delay, "message": message}


def _demo(flight, now):
    if flight not in DEMO:
        return _result(flight, "unknown", None, 0, f"{flight} · NOT IN DEMO SCHEDULE")
    gate, new_gate, delay = DEMO[flight]
    phase = PHASES[(int(now // STEP_SECONDS) + list(DEMO).index(flight)) % len(PHASES)]
    if phase in ("scheduled", "boarding"):
        return _result(flight, phase, gate, 0, _message(flight, phase, gate, 0))
    if phase == "gate_change":
        return _result(flight, "boarding", new_gate, 0, _message(flight, "boarding", new_gate, 0, True))
    if phase == "delayed":
        return _result(flight, "delayed", new_gate, delay, _message(flight, "delayed", new_gate, delay, True))
    return _result(flight, "departed", new_gate, delay, _message(flight, "departed", new_gate, delay))


def _live(flight):
    url = "http://api.aviationstack.com/v1/flights?" + urllib.parse.urlencode({"access_key": KEY, "flight_iata": flight})
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = json.load(response)
        if "error" in body:
            raise RuntimeError(body["error"].get("message") or body["error"])
        data = body.get("data") or []
        if not data:
            return _result(flight, "unknown", None, 0, f"{flight} · NO DATA")
        info = data[0]
        dep = info.get("departure") or {}
        raw = info.get("flight_status") or "unknown"
        status = {"active": "departed", "incident": "unknown", "diverted": "unknown"}.get(raw, raw)
        if status not in ("scheduled", "departed", "landed", "cancelled"):
            status = "unknown"
        delay = int(dep.get("delay") or 0)
        if status == "scheduled" and delay > 0:
            status = "delayed"
        gate = dep.get("gate")
        return _result(flight, status, gate, delay, _message(flight, status, gate, delay))
    except Exception as error:  # the HUD must keep running when the network or API fails
        return _result(flight, "unknown", None, 0, f"{flight} · STATUS UNAVAILABLE ({error})")


def status(number: str) -> dict:
    flight = number.replace(" ", "").upper()
    return _live(flight) if KEY else _demo(flight, time.time())


def selftest():
    seen = set()
    for step in range(len(PHASES)):
        r = _demo("UA123", step * STEP_SECONDS)
        assert set(r) == {"flight", "status", "gate", "delay_minutes", "message"}, r
        assert r["flight"] == "UA123" and isinstance(r["gate"], str) and isinstance(r["delay_minutes"], int)
        seen.add(r["status"])
        print(r["message"])
    assert seen == {"scheduled", "boarding", "delayed", "departed"}, seen
    assert _demo("UA123", 3 * STEP_SECONDS)["message"] == "UA123 · GATE CHANGED TO B12 · DELAYED 25 MIN"
    assert _demo("ZZ999", 0)["status"] == "unknown"
    assert status("zz 999")["status"] == "unknown"
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif len(sys.argv) == 2:
        print(json.dumps(status(sys.argv[1]), ensure_ascii=False))
    else:
        sys.exit("usage: python flight.py UA123 | --selftest")
