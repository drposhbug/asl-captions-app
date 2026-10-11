<script def>
{
  "navigationBarTitleText": "Live Captions"
}
</script>

<script setup>
import wx from 'wx';
import { LanguageModel } from 'language-model';
import { GlossStream, buildPrompt, parseSentence, matchPhrase, SYSTEM_PROMPT, AUTO_SPEAK_CONFIDENCE } from './signing.js';
import { fallbackSentence } from './dictionary.js';

// A fresh look at the scene costs an image prompt, so take one at most this often.
const SCENE_REFRESH_MS = 15000;
// Reported to the laptop on start so the build running on the glasses can be confirmed remotely.
const BUILD = 'b22-feed-on';

// shortcut: line width is estimated by character count (about 33 fit at 30px); measure text if captions clip.
const MAX_CHARS_PER_LINE = 28;
const MAX_LINES = 4;
// The laptop running server/hands_server.py; the glasses and laptop must share a network.
// The laptop's address on each Wi-Fi we use (hackathon, Verizon hotspot); the first one whose hub answers wins.
const LAPTOP_IPS = ['192.168.137.1', '172.16.240.208', '192.168.0.172', '172.16.241.181'];
let laptop = LAPTOP_IPS[0];
const hubUrl = () => `http://${laptop}:8766`;
const clipUrl = () => `http://${laptop}:8765/clips`;
const handUrl = () => `http://${laptop}:8765/hands`;
// One low-quality photo per PHOTO_MS goes to the hand server; slower than video but the only capture that works here.
const PHOTO_MS = 100;

// Probes every candidate hub in parallel; resolves true when one answers so the page can start posting.
function findLaptop() {
  return new Promise((resolve) => {
    let pending = LAPTOP_IPS.length;
    LAPTOP_IPS.forEach((ip) => {
      wx.request({
        url: `http://${ip}:8766/events?since=0`, timeout: 2500, responseType: 'text',
        success: (res) => { if (res.statusCode === 200) { laptop = ip; resolve(true); } },
        complete: () => { pending -= 1; if (!pending) resolve(false); },
      });
    });
  });
}
const MAX_CLIP_MS = 60000;
// shortcut: maps the whole camera frame onto the display; the camera sees far wider than the display,
// so the skeleton won't line up with your real hands. Raise above 1 (crop to centre) to calibrate on device.
const OVERLAY_ZOOM = 1;
// The server's 21 hand landmarks (MediaPipe order): wrist, then four joints per finger from thumb to pinky.
// shortcut: hardcoded wearer name; read it from settings once the agent has a settings page.
const USER_NAME = 'Oscar';
const NAME_PATTERN = new RegExp(`\\b${USER_NAME}\\b`, 'i');
// Saying "Jarvis" wakes the app like "Hi Rokid" (that system wake word can't be renamed); ASR often hears "Jervis".
const WAKE_PATTERN = /\bj[ae]rvis\b/i;
// An alert flashes for FLASH_MS to grab attention, then holds steady so it can be read, until BANNER_MS.
const FLASH_MS = 3000;
const BANNER_MS = 8000;
// A quick phrase stays armed for the next tap this long after the last swipe.
const QUICK_PHRASE_MS = 8000;
// A nod or shake is only an answer if someone asked something this recently (or a sentence is waiting on screen).
const QUESTION_WINDOW_MS = 10000;
const QUESTION_RE = /\?\s*$|^(do|does|did|are|is|can|could|would|will|should|have|has|was|were|what|where|when|who|why|how|any|anything)\b/i;
const QUICK_PHRASES = [
  "I'm deaf, please face me so I can read you.",
  'Please write it down or type it on your phone.',
  'I use sign language. Please speak slowly and clearly.',
  'Can you point me to the gate?',
  'I need help, please call someone who can sign.',
];
const ALERT_TEXT = {
  fire_alarm: '! FIRE ALARM',
  siren: '! SIREN',
  car_horn: '! CAR HORN',
  door_knock: '! DOOR KNOCK',
};
const LABEL_PROMPT = 'Given what the wearer sees and what was just said, answer with ONE label in caps for who is speaking: e.g. TSA AGENT, GATE AGENT, ANNOUNCEMENT, PILOT, CASHIER, DOCTOR, FRIEND, PERSON. Only the label.';
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

// Camera calls can hang when no camera answers; fail visibly instead of showing "recording" forever.
function withTimeout(promise, ms, what) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${what} timed out`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function openCamera(width, height) {
  // Video only: the microphone stays with captions.
  const request = navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { width: { ideal: width }, height: { ideal: height }, frameRate: { ideal: 30 } },
  });
  // A stream that arrives after the timeout must still be released.
  let timedOut = false;
  request.then((late) => { if (timedOut) late.getTracks().forEach((track) => track.stop()); }, () => {});
  try {
    return await withTimeout(request, 10000, 'Camera');
  } catch (error) {
    timedOut = true;
    throw error;
  }
}

function releaseCamera(run) {
  if (run.stream) run.stream.getTracks().forEach((track) => track.stop());
  run.stream = null;
}

// The renderer has no word-break, so wrap between words ourselves and keep the newest lines.
function toLines(text) {
  const lines = [];
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const last = lines[lines.length - 1];
    if (last && last.length + 1 + word.length <= MAX_CHARS_PER_LINE) lines[lines.length - 1] = `${last} ${word}`;
    else lines.push(word);
  }
  return lines.slice(-MAX_LINES);
}

// Blob.arrayBuffer() is undocumented in this runtime; Response.arrayBuffer() is, so try both.
// fetch() on the glasses gives up after 5 s, too short for photos and clips on slow Wi-Fi;
// wx.request takes an explicit timeout. Resolves with the JSON reply ({} for an allowed error status).
async function postBinary(url, body, type, timeout, allowStatus = []) {
  const data = body instanceof ArrayBuffer ? body
    : typeof body.arrayBuffer === 'function' ? await body.arrayBuffer() : await new Response(body).arrayBuffer();
  return new Promise((resolve, reject) => {
    wx.request({
      url, method: 'POST', data, timeout, header: { 'Content-Type': type }, responseType: 'text', dataType: 'json',
      success: (res) => {
        if (res.statusCode >= 200 && res.statusCode < 300) resolve(typeof res.data === 'string' ? JSON.parse(res.data) : res.data);
        else if (allowStatus.includes(res.statusCode)) resolve({});
        else reject(new Error(`Server replied ${res.statusCode}`));
      },
      fail: (err) => reject(new Error((err && err.errMsg) || 'Request failed')),
    });
  });
}

// Frames are a Blob (video grab) or an ArrayBuffer (photo camera fallback).
async function blobToBase64(blob) {
  const buffer = blob instanceof ArrayBuffer ? blob
    : typeof blob.arrayBuffer === 'function' ? await blob.arrayBuffer() : await new Response(blob).arrayBuffer();
  return wx.arrayBufferToBase64(buffer);
}

// The hub is a nice-to-have log; it must never slow down or break the glasses UI.
function postEvent(type, text, confidence = null, label = null) {
  fetch(`${hubUrl()}/events`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type, text, confidence, label, at: Date.now() }),
  }).catch(() => {});
}

// The runtime reports "view not active" as InvalidStateError, sometimes only inside the message text.
function isNotInteractive(error) {
  const text = error ? `${error.name} ${error.error} ${error.message}` : '';
  return /InvalidStateError|blurred|non-?interactive|not interactive/i.test(text);
}

export default {
  data: {
    listening: false,
    lines: [],
    error: '',
    clipStatus: '',
    handStatus: '',
    laptopStatus: '',
    glossText: '',
    sentence: '',
    sentenceStatus: '',
    sentencePercent: 0,
    banner: '',
    // Degrees the alert's sound came from (-90 left .. 0 ahead .. 90 right), or null: rotates the banner arrow.
    bannerAngle: null,
    bannerWhere: '',
    flash: false,
    quickPhrase: '',
    canvasWidth: 480,
    canvasHeight: 352,
  },

  onLoad() {
    this.recognition = null;
    this.segments = new Map();
    this.utterance = []; // segment keys of the utterance being heard, cleaned together when it ends
    this.polished = []; // cleaned lines, sent back as context for the next cleanup
    // Per caption segment: recognition confidence and speaker label.
    this.marks = new Map();
    this.lastKey = null;
    this.quickIndex = -1;
    this.glossStream = new GlossStream();
    this.pendingSentence = null;
    this.scene = '';
    this.sceneAt = 0;
    // Head gestures are optional signals; the tap path below stays usable without them.
    if (typeof this.enableWorldAwareness === 'function') this.enableWorldAwareness();
    this.setData({ canvasWidth: window.innerWidth, canvasHeight: window.innerHeight });
  },

  onShow() {
    this.visible = true;
    this.startCaptions();
    if (!this.alertTimer) this.alertTimer = setInterval(() => this.pollAlerts(), 500);
    // Who the laptop hears talking (enrolled voices), refreshed every 2 s; unknown voices stay unlabelled.
    if (!this.speakerTimer) {
      this.speakerTimer = setInterval(() => {
        fetch(`${hubUrl()}/speaker`)
          .then((response) => (response.ok ? response.json() : null))
          .then((body) => {
            // Name (enrolled voice) and side (stereo mic) are independent; either alone is worth showing.
            if (body && (body.name || body.side)) this.knownSpeaker = { name: body.name || '', side: body.side || '', at: Date.now() };
          })
          .catch(() => {});
      }, 2000);
    }
    // Captions from the laptop's mic (hub + ElevenLabs): a second ear for talkers too quiet for the glasses.
    if (!this.captionTimer) this.captionTimer = setInterval(() => this.pollCaptions(), 500);
    // onHostFocus does not always fire; keep trying to start the camera until it opens or is switched off.
    if (!this.handsTimer) {
      this.handsTimer = setInterval(() => {
        if (this.visible && !this.handRun && !this.recording && !this.handsOff) this.startHands();
      }, 5000);
    }
    // Re-probe until the laptop answers, and again every 30 s in case the Wi-Fi changed.
    if (!this.laptopTimer) this.laptopTimer = setInterval(() => findLaptop(), 30000);
    findLaptop().then((found) => {
      this.setData({ laptopStatus: found ? `LAPTOP ${laptop}` : 'LAPTOP NOT FOUND' });
      if (found) postEvent('status', `glasses build ${BUILD}`);
      // One spoken line on connect: tests the ElevenLabs voice (glasses + laptop) without a tap.
      if (found && !this.voiceTested) {
        this.voiceTested = true;
        this.speakSentence('Jarvis online. ElevenLabs voice test.', 100);
      }
    });
    // Rough location as caption-cleanup context (an airport, a cafe); denial just means no location.
    if (this.locationWatch == null && typeof navigator !== 'undefined' && navigator.geolocation) {
      this.locationWatch = navigator.geolocation.watchPosition(
        (position) => {
          const { latitude, longitude, accuracy } = position.coords;
          this.location = `${latitude.toFixed(4)}, ${longitude.toFixed(4)} (within ${Math.round(accuracy)} m)`;
        },
        () => {},
        { enableHighAccuracy: false, maximumAge: 60000, timeout: 20000 },
      );
    }
  },

  onHide() {
    this.visible = false;
    this.stopCaptions();
    this.stopHands();
    this.stopRecording();
    if (this.alertTimer) clearInterval(this.alertTimer);
    this.alertTimer = null;
    if (this.handsTimer) clearInterval(this.handsTimer);
    this.handsTimer = null;
    if (this.speakerTimer) clearInterval(this.speakerTimer);
    this.speakerTimer = null;
    if (this.captionTimer) clearInterval(this.captionTimer);
    this.captionTimer = null;
    if (this.laptopTimer) clearInterval(this.laptopTimer);
    this.laptopTimer = null;
    if (this.locationWatch != null) navigator.geolocation.clearWatch(this.locationWatch);
    this.locationWatch = null;
  },

  // Wakeup ("Hi Rokid" or touch) belongs to captions here, not the system assistant.
  onVoiceWakeup(event) {
    event.preventDefault();
    this.wake();
  },

  // "Hi Rokid", touch wakeup or "Jarvis": captions and sign tracking on, ready for the signer.
  wake() {
    if (!this.data.listening) this.startCaptions();
    if (!this.handRun && !this.recording) {
      this.handsOff = false;
      this.startHands();
    }
    this.setData({ sentenceStatus: 'JARVIS - READY, SIGN NOW' });
    postEvent('status', 'wake: jarvis');
  },

  // SpeechRecognition.start() throws while the view is blurred; begin once it gains host focus.
  onHostFocus() {
    this.hostFocused = true;
    if (this.visible && !this.data.listening) this.startCaptions();
    // Hand tracking starts by itself once the view is interactive (the camera refuses to open before that);
    // swipe down turns it off, and off it stays until the next swipe.
    if (this.visible && !this.handRun && !this.recording && !this.handsOff) this.startHands();
  },

  onHostBlur() {
    this.hostFocused = false;
  },

  onUnload() {
    this.visible = false;
    this.stopCaptions();
    this.stopHands();
    this.stopRecording();
    if (this.alertTimer) clearInterval(this.alertTimer);
    this.alertTimer = null;
    if (this.handsTimer) clearInterval(this.handsTimer);
    this.handsTimer = null;
    if (this.speakerTimer) clearInterval(this.speakerTimer);
    this.speakerTimer = null;
    if (this.captionTimer) clearInterval(this.captionTimer);
    this.captionTimer = null;
    if (this.laptopTimer) clearInterval(this.laptopTimer);
    this.laptopTimer = null;
    if (this.locationWatch != null) navigator.geolocation.clearWatch(this.locationWatch);
    this.locationWatch = null;
    if (this.bannerTimer) clearTimeout(this.bannerTimer);
    if (this.steadyTimer) clearTimeout(this.steadyTimer);
    if (this.quickTimer) clearTimeout(this.quickTimer);
    if (this.grammar) this.grammar.destroy();
    this.grammar = null;
    if (this.labeler) this.labeler.destroy();
    this.labeler = null;
  },

  startCaptions() {
    if (typeof SpeechRecognition === 'undefined') {
      this.setData({ error: 'Speech recognition is unavailable' });
      return;
    }
    if (!this.recognition) {
      const recognition = new SpeechRecognition();
      // shortcut: English only; an empty lang lets the host guess and it starts in Chinese. Make selectable if other languages are needed.
      recognition.lang = 'en-US';
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.onstart = () => this.setData({ listening: true, error: '' });
      recognition.onresult = (event) => this.showResult(event);
      recognition.onerror = (event) => {
        if (isNotInteractive(event)) return this.setData({ listening: false });
        const message = (event && (event.message || event.error)) || 'Speech error';
        // "Cloud ASR can't accept audio" is a transient hiccup (mic busy or a dropped upload): let onend retry.
        // Also socket/timeout/abort: the glasses' internet dropped; keep retrying until it is back.
        if (/accept audio|queue|full|audio-capture|network|no-speech|socket|timeout|abort/i.test(message)) {
          postEvent('status', `speech: ${message}`);
          return this.setData({ listening: false });
        }
        this.setData({ error: message });
      };
      // The host may end a session after each utterance; keep captions running while visible.
      recognition.onend = () => {
        // The session's last segment gets no successor index, so it is final here.
        if (this.lastKey) this.finalizeSegment(this.lastKey);
        this.lastKey = null;
        this.polishUtterance();
        this.setData({ listening: false });
        // A short pause before restarting lets the cloud ASR drain its audio queue instead of overflowing it.
        if (this.captionRestart) clearTimeout(this.captionRestart);
        this.captionRestart = setTimeout(() => {
          if (this.visible && !this.data.error && !this.data.listening && !this.uploading) this.startCaptions();
        }, 400);
      };
      this.recognition = recognition;
    }
    try {
      this.recognition.start();
    } catch (error) {
      // Not interactive yet: wait for onHostFocus instead of showing an error.
      if (isNotInteractive(error)) {
        this.setData({ listening: false });
        return;
      }
      this.setData({ listening: false, error: String((error && error.message) || error) });
    }
  },

  stopCaptions() {
    if (!this.recognition) return;
    const recognition = this.recognition;
    this.recognition = null;
    try {
      recognition.abort();
    } catch (_) {}
    this.setData({ listening: false });
  },

  showResult(event) {
    const result = event && event.results && event.results[event.resultIndex];
    const transcript = result && result[0] && result[0].transcript;
    // The host sometimes ignores lang='en-US' and answers in Chinese; drop those results.
    if (!transcript || /[\u3400-\u9FFF]/.test(transcript)) return;
    // Interim and final results for the same index replace each other.
    const key = `${event.sessionId}:${event.resultIndex}`;
    this.segments.set(key, transcript.trim());
    this.lastHeard = { text: transcript.trim(), at: Date.now() };
    // Confidence is undocumented here; when it is missing, leave the line unmarked.
    const mark = this.marks.get(key) || {};
    mark.confidence = typeof result[0].confidence === 'number' ? result[0].confidence : null;
    this.marks.set(key, mark);
    // No isFinal in this runtime: a segment is final once the next index (or session) starts.
    if (this.lastKey && this.lastKey !== key) this.finalizeSegment(this.lastKey);
    this.lastKey = key;
    if (!this.utterance.includes(key)) this.utterance.push(key);
    this.renderCaptions();
  },

  // The host ends a recognition session after each utterance: clean the whole utterance in one Claude call
  // (fixes misheard words and the duplicate guesses split across pieces), then show the cleaned line.
  polishUtterance() {
    const keys = this.utterance.filter((k) => this.segments.has(k));
    this.utterance = [];
    const text = keys.map((k) => this.segments.get(k)).join(' ').trim();
    if (!text) return;
    const context = this.polished.slice(-3);
    fetch(`${hubUrl()}/polish`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, context, location: this.location || null }),
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (!body || !body.text || !keys.every((k) => this.segments.has(k))) return;
        this.segments.set(keys[0], body.text);
        keys.slice(1).forEach((k) => this.segments.delete(k));
        this.polished.push(body.text);
        this.renderCaptions();
      })
      .catch(() => {});
  },

  renderCaptions() {
    const text = Array.from(this.segments, ([key, said]) => {
      const { label } = this.marks.get(key) || {};
      return `${label ? `${label}: ` : ''}${said}`;
    }).join(' ');
    this.setData({ lines: toLines(text) });
  },

  finalizeSegment(key) {
    const said = this.segments.get(key);
    if (!said) return;
    const mark = this.marks.get(key) || {};
    if (WAKE_PATTERN.test(said)) this.wake();
    if (NAME_PATTERN.test(said)) {
      this.showBanner('! SOMEONE SAID YOUR NAME');
      postEvent('alert', 'SOMEONE SAID YOUR NAME', mark.confidence, 'name');
    }
    // A voice the laptop recognises (enrolled on the dashboard) names the speaker outright.
    if (this.knownSpeaker && Date.now() - this.knownSpeaker.at < 6000) {
      const arrow = { left: '<<< ', right: '>>> ', center: '^ ' }[this.knownSpeaker.side] || '';
      mark.label = `${arrow}${this.knownSpeaker.name}`.trim();
      this.renderCaptions();
      return postEvent('caption', said, mark.confidence, mark.label);
    }
    // One label request at a time; segments arriving meanwhile go unlabelled rather than queueing up.
    if (!this.scene || this.labeling) return postEvent('caption', said, mark.confidence);
    this.labeling = true;
    this.labelSpeaker(said)
      .then((label) => {
        if (label) {
          mark.label = label;
          this.renderCaptions();
        }
        postEvent('caption', said, mark.confidence, label);
      }, () => postEvent('caption', said, mark.confidence))
      .finally(() => { this.labeling = false; });
  },

  // Its own session so labels never mix into the signing grammar's conversation.
  async labelSpeaker(said) {
    if (!this.labeler) {
      if ((await LanguageModel.availability()) !== 'available') return null;
      this.labeler = await LanguageModel.create({ initialPrompts: [{ role: 'system', content: LABEL_PROMPT }] });
    }
    const reply = await this.labeler.prompt(`Scene: ${this.scene}\nSaid: ${said}`);
    return reply.trim().split('\n')[0].toUpperCase().slice(0, 20) || null;
  },

  // One request in flight; a failed poll just waits for the next tick. A request that hangs (hub restarted
  // mid-request) is abandoned after 5 s, else alerts would stop for good.
  pollAlerts() {
    if (this.alertPolling && Date.now() - this.alertPolling < 5000) return;
    const started = Date.now();
    this.alertPolling = started;
    fetch(`${hubUrl()}/alerts?since=${this.alertSince || 0}`)
      .then((response) => {
        if (!response.ok) throw new Error(`alerts replied ${response.status}`);
        return response.json();
      })
      .then((body) => {
        if (!body || this.alertPolling !== started) return;
        // The first poll only catches up, so a stale alarm from before the page opened is not shown.
        const primed = this.alertSince !== undefined;
        this.alertSince = body.next;
        const latest = body.alerts && body.alerts[body.alerts.length - 1];
        if (latest) postEvent('status', `alert poll: ${body.alerts.length} new, primed ${primed}, kind ${latest.kind}`);
        // kind "say" is the dashboard's demo fallback: speak the line instead of bannering it
        if (primed && latest && latest.kind === 'say') this.speakSentence(latest.label, 100);
        // kind "record" comes from the dashboard/phone page posting /record: start or stop a clip
        else if (primed && latest && latest.kind === 'record') this.toggleRecording();
        else if (primed && latest) {
          const text = ALERT_TEXT[latest.kind] || `! ${String(latest.kind).toUpperCase()}`;
          // A stereo mic on the laptop (or a chest camera) gives the exact angle; older hubs only the side.
          if (typeof latest.angle === 'number') this.showBanner(text, latest.angle);
          else {
            const arrow = { left: '<<< ', right: '', center: '^ ' }[latest.direction];
            this.showBanner(latest.direction === 'right' ? `${text} >>>` : `${arrow || ''}${text}`);
          }
        }
      })
      .catch((error) => {
        // Report a failing poll to the laptop at most every 30 s so the cause is visible remotely.
        if (!this.pollErrorAt || Date.now() - this.pollErrorAt > 30000) {
          this.pollErrorAt = Date.now();
          postEvent('status', `alert poll failed: ${(error && error.message) || error}`);
        }
      })
      .finally(() => { if (this.alertPolling === started) this.alertPolling = 0; });
  },

  pollCaptions() {
    if (this.captionPolling && Date.now() - this.captionPolling < 5000) return;
    const started = Date.now();
    this.captionPolling = started;
    fetch(`${hubUrl()}/captions?since=${this.captionSince || 0}`)
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (!body || this.captionPolling !== started) return;
        const primed = this.captionSince !== undefined; // the first poll only catches up
        this.captionSince = body.next;
        if (!primed) return;
        for (const caption of body.captions || []) {
          if (!caption.text || !caption.text.trim()) continue;
          // The glasses usually hear it themselves; the laptop only fills in lines they missed.
          if (this.sameLine(caption.text)) continue;
          const key = `laptop:${caption.id}`;
          this.segments.set(key, caption.text);
          const mark = this.marks.get(key) || { confidence: null };
          mark.label = caption.label || mark.label || '';
          this.marks.set(key, mark);
        }
        this.renderCaptions();
      })
      .catch(() => {})
      .finally(() => { if (this.captionPolling === started) this.captionPolling = 0; });
  },

  // The glasses usually hear the same sentence themselves: the key of a recent line that mostly matches, else null.
  sameLine(text) {
    const words = (t) => new Set((t || '').toLowerCase().match(/[a-z0-9']+/g) || []);
    const mine = words(text);
    if (!mine.size) return null;
    const hit = Array.from(this.segments.entries()).slice(-3).find(([, said]) => {
      const theirs = words(said);
      let shared = 0;
      mine.forEach((w) => { if (theirs.has(w)) shared += 1; });
      return shared / mine.size >= 0.6;
    });
    return hit ? hit[0] : null;
  },

  // An alarm takes over the whole display and flashes; a Deaf wearer must not be able to miss it.
  showBanner(banner, angle = null) {
    if (this.bannerTimer) clearTimeout(this.bannerTimer);
    if (this.steadyTimer) clearTimeout(this.steadyTimer);
    if (this.flashTimer) clearInterval(this.flashTimer);
    const degrees = angle === null ? 0 : Math.round(Math.abs(angle));
    const bannerWhere = angle === null ? '' : degrees <= 10 ? 'AHEAD' : `${degrees}\u00B0 ${angle < 0 ? 'LEFT' : 'RIGHT'}`;
    // The banner is the only thing on screen: drop the sentence box and quick phrase under it.
    this.pendingSentence = null;
    this.setData({ banner, bannerAngle: angle === null ? null : Math.round(angle), bannerWhere, flash: true, sentence: '', glossText: '', sentenceStatus: '', quickPhrase: '' });
    this.flashTimer = setInterval(() => this.setData({ flash: !this.data.flash }), 250);
    this.steadyTimer = setTimeout(() => {
      if (this.flashTimer) clearInterval(this.flashTimer);
      this.flashTimer = null;
      this.setData({ flash: false });
    }, FLASH_MS);
    this.bannerTimer = setTimeout(() => {
      if (this.flashTimer) clearInterval(this.flashTimer);
      this.setData({ banner: '', bannerAngle: null, bannerWhere: '', flash: false });
    }, BANNER_MS);
    postEvent('status', `alert shown on glasses: ${banner}${bannerWhere ? ` ${bannerWhere}` : ''}`);
  },

  nextQuickPhrase() {
    this.quickIndex = (this.quickIndex + 1) % QUICK_PHRASES.length;
    if (this.quickTimer) clearTimeout(this.quickTimer);
    this.setData({ quickPhrase: QUICK_PHRASES[this.quickIndex] });
    this.quickTimer = setTimeout(() => this.setData({ quickPhrase: '' }), QUICK_PHRASE_MS);
  },

  // Glasses have no touchscreen: temple tap (Enter) speaks the shown quick phrase or pending sentence, else stops a
  // remotely started clip, else speaks the first quick phrase; ArrowUp cycles quick phrases (or retries captions while an error is shown); ArrowDown toggles hands.
  // ASL yes/no lives in the head: a nod confirms what is on screen (or just says "Yes"), a shake rejects it (or "No").
  onHeadGesture(event) {
    const gesture = event && event.gesture;
    if (gesture === 'nod') {
      if (this.confirmShown()) return;
      if (this.heardQuestion()) this.speakSentence('Yes.', 100);
    } else if (gesture === 'shake') {
      if (this.data.quickPhrase || this.pendingSentence) this.dismissShown();
      else if (this.heardQuestion()) this.speakSentence('No.', 100);
    }
    // Otherwise the wearer is just looking around: say nothing.
  },

  heardQuestion() {
    const heard = this.lastHeard;
    return Boolean(heard && Date.now() - heard.at < QUESTION_WINDOW_MS && QUESTION_RE.test(heard.text));
  },

  // Speaks the quick phrase or pending sentence on screen; returns false when there is none.
  confirmShown() {
    if (this.data.quickPhrase) {
      if (this.quickTimer) clearTimeout(this.quickTimer);
      const phrase = this.data.quickPhrase;
      this.setData({ quickPhrase: '' });
      this.speakSentence(phrase, 100);
    // A sentence the model was unsure of waits for the signer's confirmation.
    } else if (this.pendingSentence) this.speakSentence(this.pendingSentence, this.pendingPercent);
    else return false;
    return true;
  },

  dismissShown() {
    if (this.quickTimer) clearTimeout(this.quickTimer);
    this.pendingSentence = null;
    this.setData({ quickPhrase: '', sentence: '', sentencePercent: 0, sentenceStatus: 'DISMISSED' });
  },

  onKeyUp(event) {
    if (event.code === 'Enter') {
      event.preventDefault();
      if (this.confirmShown()) return;
      if (this.recorder) this.stopRecording();
      else this.speakSentence(QUICK_PHRASES[0], 100);
    } else if (event.code === 'ArrowUp') {
      event.preventDefault();
      if (this.data.error) this.retryCaptions();
      else this.nextQuickPhrase();
    } else if (event.code === 'ArrowDown') {
      event.preventDefault();
      this.toggleHands();
    }
  },

  retryCaptions() {
    this.setData({ error: '' });
    this.stopCaptions();
    this.startCaptions();
  },

  toggleRecording() {
    if (this.recorder) this.stopRecording();
    else this.recordClip();
  },

  stopRecording() {
    if (this.recorder && this.recorder.state !== 'inactive') this.recorder.stop();
  },

  // Records a clean clip (no overlay) and uploads it to the laptop for HaWoR and Roboflow.
  async recordClip() {
    if (this.recording) return;
    this.recording = true;
    this.stopHands(); // one camera user at a time
    this.setData({ clipStatus: 'REC - STARTING CAMERA...' });
    let stream = null;
    let limit = null;
    try {
      // Full resolution: these frames become training data.
      stream = await openCamera(1280, 720);
      if (!this.visible) throw new Error('Glasses went to sleep');
      // isTypeSupported and Blob are missing on some firmware: guard the check and record one chunk.
      const canCheck = typeof MediaRecorder.isTypeSupported === 'function';
      const mimeType = ['video/mp4', 'video/webm;codecs=vp8,opus'].find((type) => !canCheck || MediaRecorder.isTypeSupported(type));
      if (!mimeType) throw new Error('No video recording format is available');
      const recorder = new MediaRecorder(stream, { mimeType });
      const chunks = [];
      recorder.ondataavailable = (event) => { if (event.data && event.data.size) chunks.push(event.data); };
      const stopped = new Promise((resolve, reject) => {
        recorder.onstop = resolve;
        recorder.onerror = (event) => reject(new Error((event && event.error && event.error.message) || 'Recorder error'));
      });
      this.recorder = recorder;
      const started = Date.now();
      recorder.start();
      limit = setTimeout(() => this.stopRecording(), MAX_CLIP_MS);
      this.setData({ clipStatus: `REC - TAP TO STOP (MAX ${MAX_CLIP_MS / 1000}s)` });
      await stopped;
      this.recorder = null;
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
      if (!chunks.length) throw new Error('Recorder produced no data');
      const type = mimeType.split(';')[0];
      const clip = chunks.length === 1 || typeof Blob !== 'function' ? chunks[0] : new Blob(chunks, { type });
      const seconds = Math.round((Date.now() - started) / 1000);
      this.setData({ clipStatus: `UPLOADING ${seconds}s CLIP - ${Math.round((clip.size || 0) / 1024)} KB...` });
      // The upload saturates the Wi-Fi; keep the mic's audio stream off it until the clip is through.
      this.uploading = true;
      this.stopCaptions();
      const { saved } = await postBinary(clipUrl(), clip, type, 120000);
      this.setData({ clipStatus: `OK - CLIP SAVED - ${saved}` });
    } catch (error) {
      const message = (error && error.message) || String(error);
      this.setData({ clipStatus: isNotInteractive(error)
        ? 'CLIP - GLASSES NOT AWAKE, TAP AGAIN IN A SECOND'
        : `X - CLIP FAILED - ${message}` });
      postEvent('status', `clip failed: ${message}`);
    } finally {
      if (limit) clearTimeout(limit);
      if (stream) stream.getTracks().forEach((track) => track.stop());
      this.recorder = null;
      this.recording = false;
      if (this.uploading) {
        this.uploading = false;
        if (this.visible) this.startCaptions();
      }
    }
  },

  toggleHands() {
    if (this.recording) return; // the camera is busy with a clip
    if (this.handRun || this.handsWanted) {
      this.handsWanted = false;
      this.handsOff = true;
      this.stopHands();
      this.setData({ handStatus: 'HANDS OFF - SWIPE DOWN TO START' });
    } else {
      this.handsOff = false;
      this.startHands();
    }
  },


  // Streams camera photos to the WiLoR hand server and draws the returned hand skeletons.
  // One photo in flight at a time; small low-quality photos leave the mic's audio upload room, so captions keep running.
  async startHands() {
    if (this.handRun) return;
    // A token per run: a stopped loop still awaiting a frame must not touch a newer run.
    const run = (this.handRun = {});
    this.handsWanted = false;
    const active = () => this.handRun === run;
    this.setData({ handStatus: 'HANDS - STARTING CAMERA...' });
    let ctx = null;
    try {
      postEvent('status', 'hands: starting camera');
      ctx = wx.createCanvasContext('hands');
      if (!ctx) throw new Error('Hand overlay canvas is missing');
      // Two capture paths: ImageCapture on the video track (no system camera round trip, so it should be much
      // faster) and the wx photo camera, proven on these glasses but ~1.3 s per shot. The first failure of the
      // track path switches to the photo camera for the rest of the run.
      const camera = wx.media && wx.media.createCameraContext();
      let capture = null;
      if (typeof ImageCapture === 'function') {
        try {
          run.stream = await openCamera(640, 480);
          capture = new ImageCapture(run.stream.getVideoTracks()[0]);
        } catch (error) {
          postEvent('status', `hands: no video track: ${(error && error.message) || error}`);
          releaseCamera(run);
        }
      }
      if (!camera && !capture) throw new Error('Camera is unavailable');
      const shoot = async () => {
        if (capture) {
          try {
            return await capture.takePhoto({ quality: 'low', enableSystemPreview: false });
          } catch (error) {
            postEvent('status', `hands: track capture failed, using photo camera: ${(error && error.message) || error}`);
            capture = null;
            releaseCamera(run);
            if (!camera) throw error;
          }
        }
        const photo = await camera.takePhoto({ quality: 'low', enableSystemPreview: false });
        if (!photo || !photo.data) throw new Error('Camera returned no image');
        return photo.data;
      };
      const started = Date.now();
      let frames = 0;
      while (active()) {
        const shotAt = Date.now();
        const photo = await shoot();
        const shotMs = Date.now() - shotAt;
        if (!active()) break;
        // 503 = the server has no hand model yet; keep streaming so the dashboard still mirrors the camera.
        const { hands = [], gloss = null } = await postBinary(handUrl(), photo, 'image/jpeg', 30000, [503]);
        if (!active()) break;
        this.drawHands(ctx, hands);
        this.onGlossFrame(gloss, photo);
        frames += 1;
        if (frames <= 3) postEvent('status', `hands: photo ${frames} via ${capture ? 'video track' : 'photo camera'} ${Math.round((photo.byteLength || photo.size || 0) / 1024)} KB, shot ${shotMs} ms, round trip ${Date.now() - shotAt} ms`);
        const seen = gloss ? `${gloss.label} ${Math.round(gloss.confidence * 100)}%` : `${hands.length} SEEN`;
        this.setData({ handStatus: `HANDS - ${seen} - ${((frames * 1000) / (Date.now() - started)).toFixed(1)} FPS` });
        const wait = PHOTO_MS - (Date.now() - shotAt);
        if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
      }
    } catch (error) {
      // Stopping releases the camera mid-frame; that failure is expected, not worth showing.
      if (active() && isNotInteractive(error)) {
        this.handsWanted = true;
        this.setData({ handStatus: 'HANDS - WAITING FOR THE GLASSES TO WAKE (TAP ONCE)' });
        if (!this.handsWaitReported) {
          this.handsWaitReported = true;
          postEvent('status', `hands: camera refused as not interactive: ${(error && error.message) || error}`);
        }
      } else if (active()) {
        const message = (error && error.message) || String(error);
        this.setData({ handStatus: `X - HANDS FAILED - ${message}` });
        postEvent('status', `hands failed: ${message}`);
      }
    } finally {
      releaseCamera(run);
      if (active()) this.handRun = null;
      if (ctx && !this.handRun) this.drawHands(ctx, []);
    }
  },

  stopHands() {
    if (!this.handRun) return;
    releaseCamera(this.handRun);
    this.handRun = null;
    this.setData({ handStatus: '' });
  },

  // Each tracked frame: debounce the gloss, and once the signer pauses, turn the glosses into English.
  onGlossFrame(gloss, jpeg) {
    const now = Date.now();
    const committed = this.glossStream.push(gloss, now);
    if (committed) {
      postEvent('sign', committed, gloss.confidence);
      this.setData({ glossText: this.glossStream.glosses.join(' ') });
      // Look at the scene when a sentence starts so the model knows what the signer means by "this".
      if (now - this.sceneAt > SCENE_REFRESH_MS) {
        this.sceneAt = now;
        if (jpeg) this.describeScene(jpeg).catch(() => {});
      }
    }
    const finished = this.glossStream.endSentence(now);
    if (finished) this.fillSentence(finished).catch((error) => this.setData({ sentenceStatus: `X - ${(error && error.message) || error}` }));
  },

  async getGrammar() {
    if (this.grammar) return this.grammar;
    if ((await LanguageModel.availability()) !== 'available') throw new Error('Language model is unavailable');
    // One session for the whole conversation: earlier sentences help disambiguate later ones.
    this.grammar = await LanguageModel.create({ initialPrompts: [{ role: 'system', content: SYSTEM_PROMPT }] });
    return this.grammar;
  },

  async describeScene(jpeg) {
    if ((await LanguageModel.availability()) !== 'available') return;
    const session = await LanguageModel.create();
    try {
      const url = `data:image/jpeg;base64,${await blobToBase64(jpeg)}`;
      this.scene = await session.prompt([{
        role: 'user',
        content: [
          { type: 'text', text: 'In under 15 words, what is the wearer of this camera looking at? Name objects, people and signs.' },
          { type: 'image_url', image_url: { url } },
        ],
      }]);
    } finally {
      session.destroy();
    }
  },

  async fillSentence(glosses) {
    this.pendingSentence = null;
    const phrase = matchPhrase(glosses);
    if (phrase) return this.speakSentence(phrase, 100);
    this.setData({ sentence: '', sentencePercent: 0, sentenceStatus: 'FILLING...' });
    const speech = Array.from(this.segments.values()).slice(-3).join(' ').slice(-300);
    let result = null;
    try {
      const grammar = await this.getGrammar();
      result = parseSentence(await grammar.prompt(buildPrompt({ glosses, speech, scene: this.scene })));
    } catch (_) {}
    // No model (offline) or it failed: a dictionary sentence beats silence, but the signer confirms it.
    const offline = !result;
    if (offline) result = { sentence: fallbackSentence(glosses), confidence: 0.5 };
    const percent = Math.round(result.confidence * 100);
    if (!offline && result.confidence >= AUTO_SPEAK_CONFIDENCE) {
      this.speakSentence(result.sentence, percent);
    } else {
      this.pendingSentence = result.sentence;
      this.pendingPercent = percent;
      const sentenceStatus = offline ? 'OFFLINE - TAP TO SAY IT' : `${percent}% SURE - TAP TO SAY IT`;
      this.setData({ sentence: result.sentence, sentencePercent: percent, sentenceStatus });
      postEvent('sentence', result.sentence, result.confidence);
    }
  },

  speakSentence(sentence, percent) {
    this.pendingSentence = null;
    this.speakAloud(sentence);
    this.setData({ glossText: '', sentence, sentencePercent: percent || 0, sentenceStatus: percent ? `SPOKEN - ${percent}% SURE` : 'SPOKEN' });
    postEvent('sentence', sentence, percent ? percent / 100 : null);
  },

  // ElevenLabs voice via the hub; any failure (hub down, no key, undecodable audio) uses the built-in voice.
  async speakAloud(sentence) {
    try {
      const response = await fetch(`${hubUrl()}/tts?text=${encodeURIComponent(sentence)}`);
      if (!response.ok) throw new Error(`tts replied ${response.status}`);
      this.audioContext = this.audioContext || new AudioContext();
      await this.audioContext.resume();
      // Raw 16 kHz 16-bit PCM straight into a buffer: no codec needed on the glasses.
      const pcm = new Int16Array(await response.arrayBuffer());
      if (!pcm.length) throw new Error('tts returned no audio');
      const buffer = this.audioContext.createBuffer(1, pcm.length, 16000);
      buffer.copyToChannel(Float32Array.from(pcm, (s) => s / 32768), 0);
      // A new line cuts the old one instead of overlapping it (rapid taps or dashboard keys).
      if (this.voiceSource) { try { this.voiceSource.stop(); } catch (_) {} }
      const source = this.audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(this.audioContext.destination);
      source.start();
      this.voiceSource = source;
      // The glasses' own mic hears the voice and captions it: keep captions off until the line has played.
      const playingMs = Math.round((pcm.length / 16000) * 1000) + 300;
      this.stopCaptions();
      if (this.captionResume) clearTimeout(this.captionResume);
      this.captionResume = setTimeout(() => { if (this.visible && !this.uploading) this.startCaptions(); }, playingMs);
      // Which voice played, so a tap test on the glasses shows whether ElevenLabs is working.
      this.setData({ sentenceStatus: `SPOKEN - ELEVENLABS (${(pcm.length / 16000).toFixed(1)} s)` });
      postEvent('status', `elevenlabs ok: ${(pcm.length / 16000).toFixed(1)} s`);
    } catch (error) {
      postEvent('status', `elevenlabs: ${(error && error.message) || error}`);
      speechSynthesis.speak(new SpeechSynthesisUtterance(sentence), 'immediate');
      this.setData({ sentenceStatus: `SPOKEN - BUILT-IN VOICE (${((error && error.message) || error)})`.slice(0, 60) });
    }
  },

  drawHands(ctx, hands) {
    const { canvasWidth: width, canvasHeight: height } = this.data;
    const toScreen = ([x, y]) => [
      ((x - 0.5) * OVERLAY_ZOOM + 0.5) * width,
      ((y - 0.5) * OVERLAY_ZOOM + 0.5) * height,
    ];
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = '#40ff5e';
    ctx.fillStyle = '#40ff5e';
    ctx.lineWidth = 3;
    for (const hand of hands) {
      const points = hand.points.map(toScreen);
      ctx.beginPath();
      for (const [from, to] of HAND_CONNECTIONS) {
        ctx.moveTo(points[from][0], points[from][1]);
        ctx.lineTo(points[to][0], points[to][1]);
      }
      ctx.stroke();
      for (const [x, y] of points) {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.flush();
  },
};
</script>

<page>
  <view class="page">
    <canvas id="hands" class="hands" width="{{canvasWidth}}" height="{{canvasHeight}}"></canvas>
    <view class="status-row" ink:if="{{!banner}}">
      <text class="status">{{ (listening ? 'LISTENING' : 'MIC OFF') + (laptopStatus ? ' - ' + laptopStatus : '') }}</text>
      <view class="actions">
        <button class="action" bindtap="nextQuickPhrase">{{ 'UP: PHRASE' }}</button>
        <button class="action" bindtap="toggleHands">{{ 'DOWN: HANDS' }}</button>
        <button class="action" bindtap="toggleRecording">{{ 'TAP - SPEAK' }}</button>
      </view>
    </view>

    <view class="signing" ink:if="{{!banner && (glossText || sentence)}}">
      <text class="gloss-text" ink:if="{{glossText}}">{{ 'SIGNED - ' + glossText }}</text>
      <text class="sentence-text" ink:if="{{sentence}}">{{sentence}}</text>
      <view class="bar" ink:if="{{sentence && sentencePercent}}"><view class="bar-fill" style="width: {{sentencePercent}}%"></view></view>
      <text class="photo-status" ink:if="{{sentenceStatus}}">{{sentenceStatus}}</text>
    </view>

    <view class="signing" ink:if="{{!banner && quickPhrase}}">
      <text class="quick-phrase">{{quickPhrase}}</text>
      <text class="photo-status">{{ 'TAP TO SAY - UP: NEXT' }}</text>
    </view>

    <view class="captions" ink:if="{{!banner}}">
      <text class="caption-text" ink:for="{{lines}}">{{item}}</text>
      <text class="caption-empty" ink:if="{{!lines.length}}">{{ listening ? 'Waiting for speech...' : 'Wake the glasses to start captions' }}</text>
    </view>

    <view class="error" ink:if="{{!banner && error}}">
      <text class="error-text">{{ '! ERROR - ' + error }}</text>
      <button class="action" bindtap="retryCaptions">{{ 'UP: RETRY' }}</button>
    </view>

    <text class="photo-status" ink:if="{{!banner && clipStatus}}">{{clipStatus}}</text>
    <text class="photo-status" ink:if="{{!banner && handStatus}}">{{handStatus}}</text>
    <!-- Last child so the alert paints above everything else. -->
    <view class="{{ flash ? 'banner banner-flash' : 'banner' }}" ink:if="{{banner}}">
      <view class="arrow" ink:if="{{bannerWhere}}" style="transform: rotate({{bannerAngle}}deg)">
        <view class="{{ flash ? 'arrow-head arrow-head-left arrow-flash' : 'arrow-head arrow-head-left' }}"></view>
        <view class="{{ flash ? 'arrow-head arrow-head-right arrow-flash' : 'arrow-head arrow-head-right' }}"></view>
        <view class="{{ flash ? 'arrow-shaft arrow-flash' : 'arrow-shaft' }}"></view>
      </view>
      <text class="{{ flash ? 'banner-text banner-text-flash' : 'banner-text' }}">{{banner}}</text>
      <text class="{{ flash ? 'banner-text banner-text-flash' : 'banner-text' }}" ink:if="{{bannerWhere}}">{{bannerWhere}}</text>
    </view>
  </view>
</page>

<style>
.page {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px 16px;
  box-sizing: border-box;
  background-color: #000000;
  color: rgba(64, 255, 94, 0.72);
  font-family: sans-serif;
}

/* Black is see-through on the glasses, so the skeleton floats over the real world.
   Full-screen layer behind the text; it ignores the page padding so landmarks map to the whole display. */
.hands {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.status-row {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
}

.status {
  font-size: 12px;
  font-weight: 500;
  color: rgba(64, 255, 94, 0.72);
}

.actions {
  display: flex;
  flex-direction: row;
  gap: 8px;
}

.action {
  flex-shrink: 0;
  height: 32px;
  padding: 0 12px;
  font-size: 12px;
  font-weight: 500;
  color: rgba(64, 255, 94, 0.72);
  background-color: #000000;
  border: 1px solid rgba(64, 255, 94, 0.48);
  border-radius: 4px;
}

.signing {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px;
  border: 1px solid rgba(64, 255, 94, 0.48);
  border-radius: 4px;
}

.gloss-text {
  font-size: 14px;
  color: rgba(64, 255, 94, 0.72);
}

/* The signer's own sentence: what the hearing person is about to hear. */
.sentence-text {
  font-size: 24px;
  font-weight: 500;
  color: #40ff5e;
}

/* Confidence of the filled sentence; empty track = unsure. */
.bar {
  height: 6px;
  border: 1px solid rgba(64, 255, 94, 0.48);
  border-radius: 4px;
}

.bar-fill {
  height: 100%;
  background-color: #40ff5e;
}

.quick-phrase {
  font-size: 20px;
  font-weight: 500;
  color: #40ff5e;
}

/* Sound alerts and name mentions must win attention over everything else on screen. */
.banner {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 16px;
  box-sizing: border-box;
  border: 6px solid #40ff5e;
  background-color: #000000;
}

/* An arrow pointing where the sound is: 0deg is straight ahead (up), rotated by the mic's angle. */
.arrow {
  position: relative;
  width: 80px;
  height: 110px;
  margin-bottom: 12px;
}

.arrow-shaft {
  position: absolute;
  left: 32px;
  top: 14px;
  width: 16px;
  height: 96px;
  background-color: #40ff5e;
}

.arrow-head {
  position: absolute;
  top: 14px;
  width: 16px;
  height: 56px;
  background-color: #40ff5e;
}

.arrow-head-left {
  left: 32px;
  transform: rotate(40deg);
  transform-origin: 50% 0;
}

.arrow-head-right {
  left: 32px;
  transform: rotate(-40deg);
  transform-origin: 50% 0;
}

.arrow-flash {
  background-color: #000000;
}

.banner-flash {
  background-color: #40ff5e;
}

.banner-text {
  font-size: 44px;
  font-weight: 700;
  color: #40ff5e;
}

.banner-text-flash {
  color: #000000;
}

.captions {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}

/* Captions are the key content: full luminance, oversized for readability. */
.caption-text {
  font-size: 30px;
  height: 38px;
  line-height: 38px;
  font-weight: 500;
  color: #40ff5e;
}

.caption-empty {
  font-size: 18px;
  color: rgba(64, 255, 94, 0.48);
}

.error {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px;
  border: 1px dashed rgba(64, 255, 94, 0.72);
  border-radius: 4px;
}

.error-text {
  font-size: 12px;
  color: rgba(64, 255, 94, 0.72);
}

.photo-status {
  font-size: 12px;
  color: rgba(64, 255, 94, 0.72);
}
</style>
