// Turns a stream of recognised sign glosses into English, using what the glasses see and hear.
// Pure logic, no runtime APIs, so `node pages/index/signing.test.mjs` can check it.

// A gloss must be seen this many consecutive frames before it counts (filters flicker between signs).
export const COMMIT_FRAMES = 3;
// No hands (or no new gloss) for this long ends the sentence.
export const SENTENCE_GAP_MS = 1500;
// Sentences the model is at least this sure of are spoken straight away; below it, the user confirms.
export const AUTO_SPEAK_CONFIDENCE = 0.9;
export const MIN_GLOSS_CONFIDENCE = 0.6;

export const SYSTEM_PROMPT = [
  'You turn American Sign Language glosses into natural spoken English for a Deaf signer talking to a hearing person.',
  'ASL grammar differs from English: topic-comment order, no articles, no "to be", time first, no tense endings.',
  'Fill in missing words, tense, articles and pronouns. Use the conversation and scene context to pick the right meaning,',
  'e.g. after "Would you like a coffee?" the glosses "YES PLEASE SMALL" mean "Yes please, a small one."',
  'Never add ideas the signer did not sign. Keep it to one short sentence, first person.',
  'Answer only with JSON: {"sentence": "...", "confidence": 0.0-1.0}. Confidence is how sure you are this is what the signer meant.',
].join(' ');

// Debounces per-frame gloss predictions into committed glosses and sentence boundaries.
export class GlossStream {
  constructor() {
    this.glosses = [];
    this.candidate = null;
    this.candidateFrames = 0;
    this.lastActivity = 0;
  }

  // Call once per frame with the server's gloss ({label, confidence} or null).
  // Returns the gloss committed on this frame, or null.
  push(gloss, now) {
    const label = gloss && gloss.confidence >= MIN_GLOSS_CONFIDENCE ? gloss.label : null;
    if (!label) {
      this.candidate = null;
      this.candidateFrames = 0;
      return null;
    }
    this.lastActivity = now;
    this.candidateFrames = label === this.candidate ? this.candidateFrames + 1 : 1;
    this.candidate = label;
    // Holding one sign across many frames is still one gloss.
    if (this.candidateFrames !== COMMIT_FRAMES || this.glosses[this.glosses.length - 1] === label) return null;
    this.glosses.push(label);
    return label;
  }

  // Returns the finished sentence's glosses (and clears them) once the signer pauses, else null.
  endSentence(now) {
    if (!this.glosses.length || now - this.lastActivity < SENTENCE_GAP_MS) return null;
    const glosses = this.glosses;
    this.glosses = [];
    return glosses;
  }
}

// The signer's own phrases: these gloss sets are spoken word for word, skipping the language model.
// Keyed by sorted glosses, since ASL order varies ("WHERE GATE C4" = "GATE C4 WHERE").
const PHRASES = [
  ['WHICH GATE THIS', 'Which gate did they say?'],
  ['ME DEAF', "I'm deaf."],
  ['THIS GATE C4', 'Is this gate C4?'],
  ['WHERE GATE C4', 'Where is gate C4?'],
  ['THIS FLIGHT DELAYED', 'Is this flight delayed?'],
  ['WHEN FOOD ARRIVE', 'When does the food arrive?'],
  ['ME LOST', "I'm lost."],
];
const phraseKey = (glosses) => glosses.join(' ').toUpperCase().replace(/\bI\b/g, 'ME').replace(/\bC 4\b/g, 'C4')
  .split(/\s+/).sort().join(' ');
const PHRASE_BY_KEY = new Map(PHRASES.map(([glosses, sentence]) => [phraseKey([glosses]), sentence]));

export function matchPhrase(glosses) {
  return PHRASE_BY_KEY.get(phraseKey(glosses)) || null;
}

export function buildPrompt({ glosses, speech, scene }) {
  const lines = [`Glosses: ${glosses.join(' ')}`];
  if (speech) lines.push(`The hearing person recently said: "${speech}"`);
  if (scene) lines.push(`The signer is looking at: ${scene}`);
  return lines.join('\n');
}

// Models sometimes wrap JSON in prose or code fences; fall back to the raw text as a low-confidence sentence.
export function parseSentence(text) {
  const match = /\{[\s\S]*\}/.exec(text || '');
  if (match) {
    try {
      const parsed = JSON.parse(match[0]);
      if (typeof parsed.sentence === 'string' && parsed.sentence.trim()) {
        const confidence = Number(parsed.confidence);
        return { sentence: parsed.sentence.trim(), confidence: Number.isFinite(confidence) ? Math.min(1, Math.max(0, confidence)) : 0.5 };
      }
    } catch (_) {}
  }
  const sentence = (text || '').replace(/```[a-z]*/g, '').trim();
  return sentence ? { sentence, confidence: 0.5 } : null;
}
