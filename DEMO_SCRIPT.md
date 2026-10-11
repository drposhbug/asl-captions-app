# Demo script: airport check-in / TSA

Fallback if sign recognition fails on stage: on the dashboard (`http://<hub>:8766`) press a key and the glasses speak the line. `?` shows the key hint. Keep this table identical to `DEMO` in `server/hud.html`.

| Key | ASL gloss | Glasses say |
|-----|-----------|-------------|
| 1 | HELLO | Hello! |
| 2 | ME DEAF | I'm Deaf. Please face me so I can read your captions. |
| 3 | MY FLIGHT GATE WHERE | Which gate does my flight leave from? |
| 4 | HERE MY ID BOARDING-PASS | Here is my ID and boarding pass. |
| 5 | BAG CHECK NEED | Do I need to check this bag? |
| 6 | YES | Yes. |
| 7 | NO | No. |
| 8 | PLEASE WRITE DOWN | Please write it down for me. |
| 9 | BATHROOM WHERE | Where is the bathroom? |
| 0 | HELP PLEASE | Can you help me, please? |
| - | THANK-YOU | Thank you. |
| = | EMERGENCY HELP CALL PERSON SIGN | I need help. Please call someone who can sign. |

Hearing side (TSA agent, speaks aloud so captions show): "Next! ID and boarding pass, please." / "Any liquids in that bag?" / "Gate B12, boarding at 3:40."

## Run order
1. Start hub: `python server/hub.py`
2. Start hand server: `python server/hands_server.py` (wilor env)
3. Open dashboard: `http://127.0.0.1:8766` on the laptop
4. Arm phone: open the hub URL on the phone, tap ARM ALERTS
5. Wake glasses: open ASL Captions, check captions appear
6. Go: sign; if a sign misses, press its key

Cut list: if short on time drop 5, 8 and 9; never drop 1, 2 and =.

## Training the signs
Signs come from a local landmark classifier (`server/signs.py`) once `server/signs/model.pkl` exists; until then the server falls back to Roboflow.
1. Hub and hand server running, glasses streaming hands. On the dashboard press `t` to open TRAIN SIGNS.
2. Pick a label (first gloss of each line, plus `NOTHING`), hold the sign in view and press RECORD 3s. Repeat for 30-50 reps per sign, varying angle and distance. Record `NOTHING` too: resting hands, scratching, pointing at things.
3. Press TRAIN (or `python server/signs.py train`). The dashboard shows held-out accuracy per sign and the most confused pairs.
4. For each confused pair: record more reps of both, or swap one for a more distinct sign in the script above. Retrain; the server picks up the new model on the next frame.

### Importing Cindy's labelled clips
Once `C:\Users\thorn\insta360_backup\labels.xlsx` has sign names typed in (uppercase, THANK-YOU style; `quality` = skip leaves a row out):
1. `python server/import_cindy.py --dry-run` prints what it would add per sign and clip, and which rows are skipped (no sign, skip, missing `*_wilor_world_filled.jsonl` in `C:\Users\thorn\asl-project\recordings\sdcard`).
2. `python server/import_cindy.py` appends the samples to `server/signs/samples.jsonl` (every 3rd observed frame; `--stride 1` for all). Observed hands outside every labelled segment of a clip become `NOTHING`.
3. Train as above: `python server/signs.py train` or `t` on the dashboard. Rerunning the import appends again, so delete the earlier lines first if you re-import.

## Network and mic setup (do this before the demo)
The phone hotspot hands the laptop a new address every session, so the glasses lose the hub. Make the laptop the hotspot instead: it is then always `192.168.137.1`, which the glasses probe first.
1. Pixel: Settings > Network > Hotspot & tethering > USB tethering ON, with the phone plugged into the laptop. The laptop gets internet through the phone.
2. Laptop: Settings > Network & internet > Mobile hotspot ON, "Share my internet connection from" = the USB tether (shows as Ethernet). Note the network name and password it shows.
3. Glasses (Hi Rokid app) and the Pixel's browser: join the laptop's hotspot. The glasses status line should read `LAPTOP 192.168.137.1`.
4. Ace Pro 2 as the mic and chest camera: USB-C to the laptop, camera in Webcam mode. Start the hub with `--mic ace` (prints the input devices it sees; with `--stereo` the direction arrows need a 2-channel device). Clip the camera to your chest facing the person you talk to.
5. Phone page: open `http://192.168.137.1:8766` on the Pixel and add it to the home screen. Its panel says the test line, fires a test alarm, starts and stops a clip on the glasses, and shows which build the glasses are running.
6. Ace Pro 2 as the hand camera (about 3 fps with two hands on the RTX 3050, no Wi-Fi upload): plug it into the laptop in webcam mode and start the hand server with `python hands_server.py --camera 1` (0 is the built-in webcam; try 1 or 2). The glasses camera stays off by default (swipe down turns it on) so nothing is drawn over the display.
7. One command for sign-to-speech on the laptop: `powershell -ExecutionPolicy Bypass -File server\start_signing.ps1 1` (camera index) or `... start_signing.ps1 C:\path\clip.mp4 none` to replay a recorded portrait clip through the whole chain. The hub starts in Tyler's voice if it is not running. Until a sign model exists, frames are tracked but nothing is spoken.
