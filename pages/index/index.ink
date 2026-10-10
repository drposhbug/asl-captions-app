<script def>
{
  "navigationBarTitleText": "Live Captions"
}
</script>

<script setup>
import wx from 'wx';
import { LanguageModel } from 'language-model';
import { GlossStream, buildPrompt, parseSentence, SYSTEM_PROMPT, AUTO_SPEAK_CONFIDENCE } from './signing.js';
import { fallbackSentence } from './dictionary.js';

// A fresh look at the scene costs an image prompt, so take one at most this often.
const SCENE_REFRESH_MS = 15000;

// shortcut: line width is estimated by character count (about 33 fit at 30px); measure text if captions clip.
const MAX_CHARS_PER_LINE = 28;
const MAX_LINES = 4;
// The laptop running server/hands_server.py; the glasses and laptop must share a network.
const SERVER_URL = 'http://172.16.241.181:8765';
// server/hub.py on the same laptop: receives the conversation log and detects sounds (alarms, sirens, knocks).
const HUB_URL = `http://172.16.241.181:8766`;
const HAND_SERVER_URL = `${SERVER_URL}/hands`;
// Clips go to the laptop for HaWoR (server/run_hawor.py) and Roboflow frames.
const CLIP_UPLOAD_URL = `${SERVER_URL}/clips`;
const MAX_CLIP_MS = 60000;
// shortcut: maps the whole camera frame onto the display; the camera sees far wider than the display,
// so the skeleton won't line up with your real hands. Raise above 1 (crop to centre) to calibrate on device.
const OVERLAY_ZOOM = 1;
// The server's 21 hand landmarks (MediaPipe order): wrist, then four joints per finger from thumb to pinky.
// shortcut: hardcoded wearer name; read it from settings once the agent has a settings page.
const USER_NAME = 'Oscar';
const NAME_PATTERN = new RegExp(`\\b${USER_NAME}\\b`, 'i');
const BANNER_MS = 6000;
// A quick phrase stays armed for the next tap this long after the last swipe.
const QUICK_PHRASE_MS = 8000;
const QUICK_PHRASES = [
  "I'm deaf, please face me so I can read you.",
  'Please write it down or type it on your phone.',
  'I use sign language. Please speak slowly and clearly.',
  'Can you point me to the gate?',
  'I need help, please call someone who can sign.',
];
const ALERT_TEXT = {
  fire_alarm: '▲ FIRE ALARM',
  siren: '▲ SIREN',
  car_horn: '▲ CAR HORN',
  door_knock: '▲ DOOR KNOCK',
};
// Below this recognition confidence (e.g. a muffled PA announcement) the caption is marked "?".
const LOW_CAPTION_CONFIDENCE = 0.6;
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
async function blobToBase64(blob) {
  const buffer = typeof blob.arrayBuffer === 'function' ? await blob.arrayBuffer() : await new Response(blob).arrayBuffer();
  return wx.arrayBufferToBase64(buffer);
}

// The hub is a nice-to-have log; it must never slow down or break the glasses UI.
function postEvent(type, text, confidence = null, label = null) {
  fetch(`${HUB_URL}/events`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type, text, confidence, label, at: Date.now() }),
  }).catch(() => {});
}

// The runtime reports "view not active" as InvalidStateError, sometimes only inside the message text.
function isNotInteractive(error) {
  const text = error ? `${error.name} ${error.error} ${error.message}` : '';
  return /InvalidStateError|blurred/i.test(text);
}

export default {
  data: {
    listening: false,
    lines: [],
    error: '',
    clipStatus: '',
    handStatus: '',
    glossText: '',
    sentence: '',
    sentenceStatus: '',
    sentencePercent: 0,
    banner: '',
    quickPhrase: '',
    canvasWidth: 480,
    canvasHeight: 352,
  },

  onLoad() {
    this.recognition = null;
    this.segments = new Map();
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
    if (!this.alertTimer) this.alertTimer = setInterval(() => this.pollAlerts(), 1000);
  },

  onHide() {
    this.visible = false;
    this.stopCaptions();
    this.stopHands();
    this.stopRecording();
    clearInterval(this.alertTimer);
    this.alertTimer = null;
  },

  // Wakeup ("Hi Rokid" or touch) belongs to captions here, not the system assistant.
  onVoiceWakeup(event) {
    event.preventDefault();
    if (!this.data.listening) this.startCaptions();
  },

  // SpeechRecognition.start() throws while the view is blurred; begin once it gains host focus.
  onHostFocus() {
    this.hostFocused = true;
    if (this.visible && !this.data.listening) this.startCaptions();
  },

  onHostBlur() {
    this.hostFocused = false;
  },

  onUnload() {
    this.visible = false;
    this.stopCaptions();
    this.stopHands();
    this.stopRecording();
    clearInterval(this.alertTimer);
    this.alertTimer = null;
    clearTimeout(this.bannerTimer);
    clearTimeout(this.quickTimer);
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
        this.setData({ error: (event && (event.message || event.error)) || 'Speech error' });
      };
      // The host may end a session after each utterance; keep captions running while visible.
      recognition.onend = () => {
        // The session's last segment gets no successor index, so it is final here.
        if (this.lastKey) this.finalizeSegment(this.lastKey);
        this.lastKey = null;
        this.setData({ listening: false });
        if (this.visible && !this.data.error) this.startCaptions();
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
    if (!transcript) return;
    // Interim and final results for the same index replace each other.
    const key = `${event.sessionId}:${event.resultIndex}`;
    this.segments.set(key, transcript.trim());
    // Confidence is undocumented here; when it is missing, leave the line unmarked.
    const mark = this.marks.get(key) || {};
    mark.confidence = typeof result[0].confidence === 'number' ? result[0].confidence : null;
    this.marks.set(key, mark);
    // No isFinal in this runtime: a segment is final once the next index (or session) starts.
    if (this.lastKey && this.lastKey !== key) this.finalizeSegment(this.lastKey);
    this.lastKey = key;
    this.renderCaptions();
  },

  renderCaptions() {
    const text = Array.from(this.segments, ([key, said]) => {
      const { confidence, label } = this.marks.get(key) || {};
      const unsure = typeof confidence === 'number' && confidence < LOW_CAPTION_CONFIDENCE;
      return `${unsure ? '?' : ''}${label ? `${label}: ` : ''}${said}`;
    }).join(' ');
    this.setData({ lines: toLines(text) });
  },

  finalizeSegment(key) {
    const said = this.segments.get(key);
    if (!said) return;
    const mark = this.marks.get(key) || {};
    if (NAME_PATTERN.test(said)) {
      this.showBanner('▲ SOMEONE SAID YOUR NAME');
      postEvent('alert', 'SOMEONE SAID YOUR NAME', mark.confidence, 'name');
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

  // One request in flight; a failed poll just waits for the next tick.
  pollAlerts() {
    if (this.alertPolling) return;
    this.alertPolling = true;
    fetch(`${HUB_URL}/alerts?since=${this.alertSince || 0}`)
      .then((response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (!body) return;
        // The first poll only catches up, so a stale alarm from before the page opened is not shown.
        const primed = this.alertSince !== undefined;
        this.alertSince = body.next;
        const latest = body.alerts && body.alerts[body.alerts.length - 1];
        // kind "say" is the dashboard's demo fallback: speak the line instead of bannering it
        if (primed && latest && latest.kind === 'say') this.speakSentence(latest.label, 100);
        else if (primed && latest) this.showBanner(ALERT_TEXT[latest.kind] || `▲ ${String(latest.kind).toUpperCase()}`);
      })
      .catch(() => {})
      .finally(() => { this.alertPolling = false; });
  },

  showBanner(banner) {
    clearTimeout(this.bannerTimer);
    this.setData({ banner });
    this.bannerTimer = setTimeout(() => this.setData({ banner: '' }), BANNER_MS);
  },

  nextQuickPhrase() {
    this.quickIndex = (this.quickIndex + 1) % QUICK_PHRASES.length;
    clearTimeout(this.quickTimer);
    this.setData({ quickPhrase: QUICK_PHRASES[this.quickIndex] });
    this.quickTimer = setTimeout(() => this.setData({ quickPhrase: '' }), QUICK_PHRASE_MS);
  },

  // Glasses have no touchscreen: temple tap (Enter) speaks the shown quick phrase or pending sentence, else
  // starts/stops a clip; ArrowUp cycles quick phrases (or retries captions while an error is shown); ArrowDown toggles hands.
  // ASL yes/no lives in the head: a nod confirms what is on screen (or just says "Yes"), a shake rejects it (or "No").
  onHeadGesture(event) {
    const gesture = event && event.gesture;
    if (gesture === 'nod') {
      if (!this.confirmShown()) this.speakSentence('Yes.', 100);
    } else if (gesture === 'shake') {
      if (this.data.quickPhrase || this.pendingSentence) this.dismissShown();
      else this.speakSentence('No.', 100);
    }
  },

  // Speaks the quick phrase or pending sentence on screen; returns false when there is none.
  confirmShown() {
    if (this.data.quickPhrase) {
      clearTimeout(this.quickTimer);
      const phrase = this.data.quickPhrase;
      this.setData({ quickPhrase: '' });
      this.speakSentence(phrase, 100);
    // A sentence the model was unsure of waits for the signer's confirmation.
    } else if (this.pendingSentence) this.speakSentence(this.pendingSentence, this.pendingPercent);
    else return false;
    return true;
  },

  dismissShown() {
    clearTimeout(this.quickTimer);
    this.pendingSentence = null;
    this.setData({ quickPhrase: '', sentence: '', sentencePercent: 0, sentenceStatus: 'DISMISSED' });
  },

  onKeyUp(event) {
    if (event.code === 'Enter') {
      event.preventDefault();
      if (!this.confirmShown()) this.toggleRecording();
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
    this.setData({ clipStatus: 'REC · STARTING CAMERA…' });
    let stream = null;
    let limit = null;
    try {
      // Full resolution: these frames become training data.
      stream = await openCamera(1280, 720);
      if (!this.visible) throw new Error('Glasses went to sleep');
      const mimeType = ['video/mp4', 'video/webm;codecs=vp8,opus'].find((type) => MediaRecorder.isTypeSupported(type));
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
      recorder.start(1000);
      limit = setTimeout(() => this.stopRecording(), MAX_CLIP_MS);
      this.setData({ clipStatus: `● REC · TAP TO STOP (MAX ${MAX_CLIP_MS / 1000}s)` });
      await stopped;
      this.recorder = null;
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
      if (!chunks.length) throw new Error('Recorder produced no data');
      const clip = new Blob(chunks, { type: mimeType.split(';')[0] });
      const seconds = Math.round((Date.now() - started) / 1000);
      this.setData({ clipStatus: `UPLOADING ${seconds}s CLIP · ${Math.round(clip.size / 1024)} KB…` });
      const response = await withTimeout(
        fetch(CLIP_UPLOAD_URL, { method: 'POST', headers: { 'Content-Type': clip.type }, body: clip }),
        60000,
        'Clip upload',
      );
      if (!response.ok) throw new Error(`Server replied ${response.status}`);
      const { saved } = await response.json();
      this.setData({ clipStatus: `✓ CLIP SAVED · ${saved}` });
    } catch (error) {
      this.setData({ clipStatus: `✕ CLIP FAILED · ${(error && error.message) || error}` });
    } finally {
      clearTimeout(limit);
      if (stream) stream.getTracks().forEach((track) => track.stop());
      this.recorder = null;
      this.recording = false;
    }
  },

  toggleHands() {
    if (this.recording) return; // the camera is busy with a clip
    if (this.handRun) this.stopHands();
    else this.startHands();
  },

  // Streams camera frames to the WiLoR hand server and draws the returned hand skeletons.
  // One frame in flight at a time, so the frame rate follows the network and server speed.
  async startHands() {
    if (this.handRun) return;
    // A token per run: a stopped loop still awaiting a frame must not touch a newer run.
    const run = (this.handRun = {});
    const active = () => this.handRun === run;
    this.setData({ handStatus: 'HANDS · STARTING CAMERA…' });
    const ctx = wx.createCanvasContext('hands');
    try {
      if (!ctx) throw new Error('Hand overlay canvas is missing');
      // Small frames keep upload and detection fast.
      run.stream = await openCamera(640, 480);
      if (!active()) return;
      const capture = new ImageCapture(run.stream.getVideoTracks()[0]);
      const started = Date.now();
      let frames = 0;
      while (active()) {
        const bitmap = await capture.grabFrame();
        let jpeg;
        try {
          jpeg = await createImageBlob(bitmap, { type: 'image/jpeg', quality: 0.6 });
        } finally {
          bitmap.close();
        }
        const response = await withTimeout(
          fetch(HAND_SERVER_URL, { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: jpeg }),
          5000,
          'Hand server',
        );
        if (!response.ok) throw new Error(`Hand server replied ${response.status}`);
        const { hands, gloss } = await response.json();
        if (!active()) break;
        this.drawHands(ctx, hands);
        this.onGlossFrame(gloss, jpeg);
        frames += 1;
        const fps = (frames * 1000) / (Date.now() - started);
        const seen = gloss ? `${gloss.label} ${Math.round(gloss.confidence * 100)}%` : `${hands.length} SEEN`;
        this.setData({ handStatus: `HANDS · ${seen} · ${fps.toFixed(1)} FPS` });
      }
    } catch (error) {
      // Stopping releases the camera mid-frame; that failure is expected, not worth showing.
      if (active()) this.setData({ handStatus: `✕ HANDS FAILED · ${(error && error.message) || error}` });
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
        this.describeScene(jpeg).catch(() => {});
      }
    }
    const finished = this.glossStream.endSentence(now);
    if (finished) this.fillSentence(finished).catch((error) => this.setData({ sentenceStatus: `✕ ${(error && error.message) || error}` }));
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
    this.setData({ sentence: '', sentencePercent: 0, sentenceStatus: 'FILLING…' });
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
      const sentenceStatus = offline ? 'OFFLINE · TAP TO SAY IT' : `${percent}% SURE · TAP TO SAY IT`;
      this.setData({ sentence: result.sentence, sentencePercent: percent, sentenceStatus });
      postEvent('sentence', result.sentence, result.confidence);
    }
  },

  speakSentence(sentence, percent) {
    this.pendingSentence = null;
    speechSynthesis.speak(new SpeechSynthesisUtterance(sentence), 'immediate');
    this.setData({ glossText: '', sentence, sentencePercent: percent || 0, sentenceStatus: percent ? `SPOKEN · ${percent}% SURE` : 'SPOKEN' });
    postEvent('sentence', sentence, percent ? percent / 100 : null);
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
    <view class="banner" ink:if="{{banner}}">
      <text class="banner-text">{{banner}}</text>
    </view>
    <view class="status-row">
      <text class="status">{{ listening ? '● LISTENING' : '○ MIC OFF' }}</text>
      <view class="actions">
        <button class="action" bindtap="nextQuickPhrase">↑ PHRASE</button>
        <button class="action" bindtap="toggleHands">↓ HANDS</button>
        <button class="action" bindtap="toggleRecording">TAP · RECORD</button>
      </view>
    </view>

    <view class="signing" ink:if="{{glossText || sentence}}">
      <text class="gloss-text" ink:if="{{glossText}}">SIGNED · {{glossText}}</text>
      <text class="sentence-text" ink:if="{{sentence}}">{{sentence}}</text>
      <view class="bar" ink:if="{{sentence && sentencePercent}}"><view class="bar-fill" style="width: {{sentencePercent}}%"></view></view>
      <text class="photo-status" ink:if="{{sentenceStatus}}">{{sentenceStatus}}</text>
    </view>

    <view class="signing" ink:if="{{quickPhrase}}">
      <text class="quick-phrase">{{quickPhrase}}</text>
      <text class="photo-status">TAP TO SAY · ↑ NEXT</text>
    </view>

    <view class="captions">
      <text class="caption-text" ink:for="{{lines}}">{{item}}</text>
      <text class="caption-empty" ink:if="{{!lines.length}}">{{ listening ? 'Waiting for speech…' : 'Wake the glasses to start captions' }}</text>
    </view>

    <view class="error" ink:if="{{error}}">
      <text class="error-text">▲ ERROR · {{error}}</text>
      <button class="action" bindtap="retryCaptions">↑ RETRY</button>
    </view>

    <text class="photo-status" ink:if="{{clipStatus}}">{{clipStatus}}</text>
    <text class="photo-status" ink:if="{{handStatus}}">{{handStatus}}</text>
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
  width: 100%;
  padding: 8px;
  box-sizing: border-box;
  border: 2px solid #40ff5e;
  border-radius: 4px;
}

.banner-text {
  font-size: 24px;
  font-weight: 500;
  color: #40ff5e;
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
