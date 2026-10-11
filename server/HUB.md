# Hub (port 8766)
Run: `python hub.py [--sounds] [--yamnet]` (Python313: `C:\Users\thorn\AppData\Local\Programs\Python\Python313\python.exe`); self-check: `python hub.py --selftest`.
`--sounds` listens on the laptop mic (loudness detector, kind "other"); add `--yamnet` for YAMNet classes (needs tensorflow, tensorflow-hub, sounddevice).
`--mic ace` picks the input device whose name contains `ace` (case-insensitive, e.g. the Insta360 Ace Pro 2 in webcam mode) instead of the default mic; it prints every input device at startup and exits with that list if nothing matches. With `--stereo` a 1-channel device falls back to mono (warning printed).
`POST /record` (no body, no `label`) -> pushes alert kind `record` (label "") plus a `status` event; the glasses toggle a clip recording. With `?label=X&seconds=3` it is still relayed to the hand server (sign samples).
`POST /events` `{"type":"caption|sign|sentence|alert|status","text":str,"confidence":float|null,"label":str|null,"at":ms|null}` -> `{"ok":true}` (ring of 500 + `events.log`).
`GET /events?since=<id>` -> `{"events":[{"id",...}],"next":<id>}`.
`GET /alerts?since=<id>` -> `{"alerts":[{"id","kind":"fire_alarm|siren|car_horn|door_knock|other","label","confidence","at"}],"next":<id>}`.
`GET /flight?number=UA123` -> `flight.status()` dict, or 503 `{"error":"flight module unavailable"}`.
`GET /` -> HUD mirror page. On a phone open `http://<laptop-ip>:8766` (`ipconfig` for the IP) to mirror and vibrate on alerts.
Firewall (admin PowerShell, once): `netsh advfirewall firewall add rule name="ASL hub" dir=in action=allow protocol=TCP localport=8766`.
`POST /test-alert` (no body) -> pushes alert kind `fire_alarm`, label `FIRE ALARM (TEST)`, confidence 1.0 (demo / phone-buzz check): `curl -X POST http://127.0.0.1:8766/test-alert`.
`POST /say` `{"text":str,"glosses":str}` -> adds a `sign` event (glosses) and pushes alert kind `say` (label = text, confidence 1.0); the glasses speak it, the HUD shows it as the sentence (no flash).
Hand server laptop-camera mode posts to `/say` too: `python hands_server.py --camera N [--rotate cw|ccw|none] [--mirror]` (wilor env). `--rotate` turns each webcam frame to portrait before tracking (default `cw`, the chest-clamped Insta360 rig the sign model is trained on; `none` for an upright webcam); `--mirror` flips it horizontally (default off). Phrases from `PHRASES` in signing.js (`server/phrases.py`, `python phrases.py --selftest`) are spoken word for word; anything else goes through `polish.py`.
Dashboard keys `1`-`9`, `0`, `-`, `=` post the 12 lines of `DEMO_SCRIPT.md` to `/say`; `?` shows the key hint.
This is the demo fallback: if sign recognition fails on stage, press the line's key.
`GET /manifest.json`, `/sw.js`, `/icon-192.png`, `/icon-512.png` -> PWA bits so Chrome offers install ("ASL Alerts").
All responses send `Access-Control-Allow-Origin: *`.
Who is talking (`speaker.py`, resemblyzer 256-d voice embeddings; needs `--sounds`; self-check `python speaker.py --selftest`): on speech (YAMNet "Speech" >= 0.5, or loudness above the room's median without `--yamnet`) the last 2 s are matched to enrolled voices (cosine >= 0.75), held 4 s, and a `status` event `speaker: NAME 0.83` is logged when the name changes (at most every 4 s).
`GET /speaker` -> `{"name":str|null,"confidence":float,"at":ms|null}` (name null + at set = unknown voice). `GET /speaker/profiles` -> `{"ALICE":3}` embedding counts.
`POST /speaker/enroll?name=ALICE&seconds=5` -> records the next N s (1-30) of mic into `speakers/ALICE.npy`, blocks, then `{"ok":true,"samples":<embeddings in profile>}`; `POST /speaker/clear?name=ALICE` deletes it. Names `[A-Z0-9_-]{1,20}` (uppercased), else 400; enroll is 503 without `--sounds`. Enroll the wearer as `YOU`.
Dashboard key `v`: VOICES panel (enroll with countdown, profile list with CLEAR, live `SPEAKING: NAME (83%)`). Install: `pip install torch --index-url https://download.pytorch.org/whl/cpu webrtcvad-wheels librosa` then `pip install --no-deps resemblyzer` (plain `resemblyzer` needs a C compiler for webrtcvad on Windows).

## Phone alert app
HUD page: ARM ALERTS button (tap once: wake lock + test buzz). Each new `/alerts` alert, or event of type `alert` (e.g. glasses "SOMEONE SAID YOUR NAME"), de-duplicated by id: full-screen green/black flash at 4 Hz for 8 s, vibrates every 3 s; tap to dismiss.
1. Hub running, then `powershell -ExecutionPolicy Bypass -File server\tunnel.ps1` -> prints `PUBLIC HUB URL:  https://<random-words>.trycloudflare.com` (new URL each run; needs `winget install --id Cloudflare.cloudflared -e`).
2. On the phone open that URL in Chrome -> menu -> Add to Home screen -> open the app -> tap ARM ALERTS.
3. Test: `curl -X POST http://127.0.0.1:8766/test-alert` -> phone flashes and buzzes.
Limit: it only buzzes while the app is open and on screen (no push notifications; Android stops JS when the app is backgrounded or the screen locks — the wake lock keeps the screen on). Over plain LAN http the page still works but cannot be installed (service worker needs https).
- `--gain <float>` multiplies the mic before speech detection (default 1.0); the hub prints `mic level: rms/floor/trigger` every 10 s so you can see whether a talker crosses the trigger.
- Offline captions: `STT_BACKEND=local` makes the hub transcribe the laptop mic with faster-whisper (base.en, CPU, ~1 s per utterance; `STT_LOCAL_MODEL=tiny.en` is 3x faster). Without the variable the hub uses ElevenLabs and falls back to local if the key is refused.
