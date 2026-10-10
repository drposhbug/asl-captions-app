<script def>
{
  "navigationBarTitleText": "Live Captions"
}
</script>

<script setup>
import wx from 'wx';

// shortcut: line width is estimated by character count (about 33 fit at 30px); measure text if captions clip.
const MAX_CHARS_PER_LINE = 28;
const MAX_LINES = 4;
const VIDEO_TEST_MS = 3000;

// Camera calls can hang when no camera answers; fail visibly instead of showing "recording" forever.
function withTimeout(promise, ms, what) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${what} timed out`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
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
    photoStatus: '',
    videoStatus: '',
  },

  onLoad() {
    this.recognition = null;
    this.segments = new Map();
  },

  onShow() {
    this.visible = true;
    this.startCaptions();
  },

  onHide() {
    this.visible = false;
    this.stopCaptions();
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
    this.segments.set(`${event.sessionId}:${event.resultIndex}`, transcript.trim());
    this.setData({ lines: toLines(Array.from(this.segments.values()).join(' ')) });
  },

  // Glasses have no touchscreen: temple tap (Enter) takes the photo, a swipe records the video test,
  // and ArrowUp retries captions while an error is shown. Swipe-to-arrow mapping is undocumented, so accept both.
  onKeyUp(event) {
    if (event.code === 'Enter') {
      event.preventDefault();
      this.takeTestPhoto();
    } else if (event.code === 'ArrowUp' && this.data.error) {
      event.preventDefault();
      this.retryCaptions();
    } else if (event.code === 'ArrowUp' || event.code === 'ArrowDown') {
      event.preventDefault();
      this.takeTestVideo();
    }
  },

  retryCaptions() {
    this.setData({ error: '' });
    this.stopCaptions();
    this.startCaptions();
  },

  async takeTestPhoto() {
    this.setData({ photoStatus: 'CAPTURING…' });
    try {
      const camera = wx.media && wx.media.createCameraContext();
      if (!camera) throw new Error('Camera is unavailable');
      const photo = await camera.takePhoto({ quality: 'high' });
      const bytes = photo && photo.data ? photo.data.byteLength : 0;
      if (!bytes) throw new Error('Camera returned no image data');
      this.setData({
        photoStatus: `✓ PHOTO OK · ${photo.mimeType || 'image'} · ${Math.round(bytes / 1024)} KB`,
      });
    } catch (error) {
      // takePhoto needs an active view; its failure there is a generic exception, so hint at the fix.
      const hint = this.hostFocused ? '' : ' · wake the glasses, then tap again';
      this.setData({ photoStatus: `✕ PHOTO FAILED · ${(error && error.message) || error}${hint}` });
    }
  },

  // Records a short clip to prove video capture works; the clip is discarded like the photo.
  async takeTestVideo() {
    if (this.recordingVideo) return;
    this.recordingVideo = true;
    this.setData({ videoStatus: `● RECORDING ${VIDEO_TEST_MS / 1000}s…` });
    let stream = null;
    try {
      // Video only: the microphone stays with captions.
      const request = navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      });
      // A stream that arrives after the timeout must still be released.
      let timedOut = false;
      request.then((late) => { if (timedOut) late.getTracks().forEach((track) => track.stop()); }, () => {});
      try {
        stream = await withTimeout(request, 10000, 'Camera');
      } catch (error) {
        timedOut = true;
        throw error;
      }
      const { width, height } = stream.getVideoTracks()[0].getSettings();
      const mimeType = ['video/mp4', 'video/webm;codecs=vp8,opus'].find((type) => MediaRecorder.isTypeSupported(type));
      if (!mimeType) throw new Error('No video recording format is available');
      const recorder = new MediaRecorder(stream, { mimeType });
      let bytes = 0;
      recorder.ondataavailable = (event) => { bytes += event.data ? event.data.size : 0; };
      const stopped = new Promise((resolve, reject) => {
        recorder.onstop = resolve;
        recorder.onerror = (event) => reject(new Error((event && event.error && event.error.message) || 'Recorder error'));
      });
      recorder.start(1000);
      await new Promise((resolve) => setTimeout(resolve, VIDEO_TEST_MS));
      recorder.stop();
      await withTimeout(stopped, 5000, 'Recorder');
      if (!bytes) throw new Error('Recorder produced no data');
      this.setData({
        videoStatus: `✓ VIDEO OK · ${width}×${height} · ${mimeType.split(';')[0]} · ${Math.round(bytes / 1024)} KB`,
      });
    } catch (error) {
      this.setData({ videoStatus: `✕ VIDEO FAILED · ${(error && error.message) || error}` });
    } finally {
      if (stream) stream.getTracks().forEach((track) => track.stop());
      this.recordingVideo = false;
    }
  },
};
</script>

<page>
  <view class="page">
    <view class="status-row">
      <text class="status">{{ listening ? '● LISTENING' : '○ MIC OFF' }}</text>
      <view class="actions">
        <button class="action" bindtap="takeTestVideo">SWIPE · VIDEO</button>
        <button class="action" bindtap="takeTestPhoto">TAP · PHOTO</button>
      </view>
    </view>

    <view class="captions">
      <text class="caption-text" ink:for="{{lines}}">{{item}}</text>
      <text class="caption-empty" ink:if="{{!lines.length}}">{{ listening ? 'Waiting for speech…' : 'Wake the glasses to start captions' }}</text>
    </view>

    <view class="error" ink:if="{{error}}">
      <text class="error-text">▲ ERROR · {{error}}</text>
      <button class="action" bindtap="retryCaptions">↑ RETRY</button>
    </view>

    <text class="photo-status" ink:if="{{photoStatus}}">{{photoStatus}}</text>
    <text class="photo-status" ink:if="{{videoStatus}}">{{videoStatus}}</text>
  </view>
</page>

<style>
.page {
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
