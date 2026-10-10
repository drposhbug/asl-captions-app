# ASL Captions

An accessibility agent for Deaf and hard-of-hearing users on Rokid Glasses.

## Capabilities

- Shows live speech-to-text captions of nearby speech in large, high-contrast text.
- Turns recognised signs (glosses from the server's Roboflow classifier) into spoken English: the host language model fills in grammar using the captions and a camera scene description (`pages/index/signing.js`); sentences it is 90% sure of are spoken aloud, others wait for a tap.
- Tracks hands with WiLoR (on a companion laptop GPU server, `server/hands_server.py`) and draws the hand skeleton over the real world; swipe down to toggle. The server can save clean frames for training a sign model in Roboflow.
- Classifies signs locally from WiLoR landmarks (`server/signs.py`, sklearn MLP); record samples and retrain from the dashboard (key `t`). Roboflow is only the fallback before a model exists.
- Tap records a clean clip and uploads it to the laptop for HaWoR (`server/run_hawor.py`) and Roboflow training frames.
- Shows a confidence bar under each filled sentence.
- Logs captions, signs, sentences and alerts to the laptop hub (`server/hub.py`, port 8766); failures never block the glasses.
- Polls the hub for sound alerts (fire alarm, siren, car horn, door knock) and shows a 6 s banner; also bannered when someone says the wearer's name.
- Labels who is speaking (e.g. "TSA AGENT:") from the camera scene and what was said, with a separate language-model session.
- Marks captions with "?" when speech recognition reports low confidence (e.g. muffled PA announcements).
- Falls back to a dictionary sentence (`pages/index/dictionary.js`) when the language model is offline; the signer taps to speak it.
- Demo fallback: number keys on the hub dashboard (`DEMO_SCRIPT.md`) post `/say`; the glasses speak the line from the alert poll.
- Swipe up cycles quick phrases ("I'm deaf, please face me…"); tap speaks the shown phrase.
- Head nod confirms the sentence or quick phrase on screen (or says "Yes"); head shake dismisses it (or says "No").
