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
