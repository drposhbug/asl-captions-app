# Hub (port 8766)
Run: `python hub.py [--sounds] [--yamnet]` (Python313: `C:\Users\thorn\AppData\Local\Programs\Python\Python313\python.exe`); self-check: `python hub.py --selftest`.
`--sounds` listens on the laptop mic (loudness detector, kind "other"); add `--yamnet` for YAMNet classes (needs tensorflow, tensorflow-hub, sounddevice).
`POST /events` `{"type":"caption|sign|sentence|alert|status","text":str,"confidence":float|null,"label":str|null,"at":ms|null}` -> `{"ok":true}` (ring of 500 + `events.log`).
`GET /events?since=<id>` -> `{"events":[{"id",...}],"next":<id>}`.
`GET /alerts?since=<id>` -> `{"alerts":[{"id","kind":"fire_alarm|siren|car_horn|door_knock|other","label","confidence","at"}],"next":<id>}`.
`GET /flight?number=UA123` -> `flight.status()` dict, or 503 `{"error":"flight module unavailable"}`.
`GET /` -> HUD mirror page. On a phone open `http://<laptop-ip>:8766` (`ipconfig` for the IP) to mirror and vibrate on alerts.
Firewall (admin PowerShell, once): `netsh advfirewall firewall add rule name="ASL hub" dir=in action=allow protocol=TCP localport=8766`.
`POST /test-alert` (no body) -> pushes alert kind `fire_alarm`, label `FIRE ALARM (TEST)`, confidence 1.0 (demo / phone-buzz check): `curl -X POST http://127.0.0.1:8766/test-alert`.
`POST /say` `{"text":str,"glosses":str}` -> adds a `sign` event (glosses) and pushes alert kind `say` (label = text, confidence 1.0); the glasses speak it, the HUD shows it as the sentence (no flash).
Dashboard keys `1`-`9`, `0`, `-`, `=` post the 12 lines of `DEMO_SCRIPT.md` to `/say`; `?` shows the key hint.
This is the demo fallback: if sign recognition fails on stage, press the line's key.
`GET /manifest.json`, `/sw.js`, `/icon-192.png`, `/icon-512.png` -> PWA bits so Chrome offers install ("ASL Alerts").
All responses send `Access-Control-Allow-Origin: *`.

## Phone alert app
HUD page: ARM ALERTS button (tap once: wake lock + test buzz). Each new `/alerts` alert, or event of type `alert` (e.g. glasses "SOMEONE SAID YOUR NAME"), de-duplicated by id: full-screen green/black flash at 4 Hz for 8 s, vibrates every 3 s; tap to dismiss.
1. Hub running, then `powershell -ExecutionPolicy Bypass -File server\tunnel.ps1` -> prints `PUBLIC HUB URL:  https://<random-words>.trycloudflare.com` (new URL each run; needs `winget install --id Cloudflare.cloudflared -e`).
2. On the phone open that URL in Chrome -> menu -> Add to Home screen -> open the app -> tap ARM ALERTS.
3. Test: `curl -X POST http://127.0.0.1:8766/test-alert` -> phone flashes and buzzes.
Limit: it only buzzes while the app is open and on screen (no push notifications; Android stops JS when the app is backgrounded or the screen locks — the wake lock keeps the screen on). Over plain LAN http the page still works but cannot be installed (service worker needs https).
