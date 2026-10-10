# Hub (port 8766)
Run: `python hub.py [--sounds] [--yamnet]` (Python313: `C:\Users\thorn\AppData\Local\Programs\Python\Python313\python.exe`); self-check: `python hub.py --selftest`.
`--sounds` listens on the laptop mic (loudness detector, kind "other"); add `--yamnet` for YAMNet classes (needs tensorflow, tensorflow-hub, sounddevice).
`POST /events` `{"type":"caption|sign|sentence|alert|status","text":str,"confidence":float|null,"label":str|null,"at":ms|null}` -> `{"ok":true}` (ring of 500 + `events.log`).
`GET /events?since=<id>` -> `{"events":[{"id",...}],"next":<id>}`.
`GET /alerts?since=<id>` -> `{"alerts":[{"id","kind":"fire_alarm|siren|car_horn|door_knock|other","label","confidence","at"}],"next":<id>}`.
`GET /flight?number=UA123` -> `flight.status()` dict, or 503 `{"error":"flight module unavailable"}`.
`GET /` -> HUD mirror page. On a phone open `http://<laptop-ip>:8766` (`ipconfig` for the IP) to mirror and vibrate on alerts.
Firewall (admin PowerShell, once): `netsh advfirewall firewall add rule name="ASL hub" dir=in action=allow protocol=TCP localport=8766`.
All responses send `Access-Control-Allow-Origin: *`.
