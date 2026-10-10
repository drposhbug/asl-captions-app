"""Sign (gloss) classification through Roboflow's hosted inference API, on the clean full frame.

Set ROBOFLOW_API_KEY and ROBOFLOW_MODEL (the model id as Roboflow shows it, e.g. "asl-signs/2").
Without them, predict() returns None and the server keeps sending landmarks only.
"""
import base64
import json
import os
import urllib.parse
import urllib.request

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
MODEL = os.environ.get("ROBOFLOW_MODEL")


def enabled():
    return bool(API_KEY and MODEL)


def predict(jpeg_bytes):
    """Returns {"label", "confidence"} for the top class, or None when unconfigured or nothing is found."""
    if not enabled():
        return None
    url = f"https://classify.roboflow.com/{MODEL}?{urllib.parse.urlencode({'api_key': API_KEY})}"
    request = urllib.request.Request(
        url, data=base64.b64encode(jpeg_bytes), headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
    except Exception as error:  # a slow or failing cloud call must not stop the landmark stream
        print(f"Roboflow error: {error}")
        return None
    predictions = result.get("predictions") or []
    if isinstance(predictions, dict):  # multi-label responses key by class name
        predictions = [{"class": name, **value} for name, value in predictions.items()]
    if not predictions:
        return None
    top = max(predictions, key=lambda p: p.get("confidence", 0))
    return {"label": top["class"], "confidence": round(float(top["confidence"]), 3)}
