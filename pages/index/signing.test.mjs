import assert from 'node:assert/strict';
import { GlossStream, buildPrompt, parseSentence, matchPhrase, COMMIT_FRAMES, SENTENCE_GAP_MS } from './signing.js';

const stream = new GlossStream();
const hello = { label: 'HELLO', confidence: 0.95 };
for (let i = 1; i < COMMIT_FRAMES; i += 1) assert.equal(stream.push(hello, i), null, 'not committed before enough frames');
assert.equal(stream.push(hello, COMMIT_FRAMES), 'HELLO', 'committed on the Nth frame');
assert.equal(stream.push(hello, COMMIT_FRAMES + 1), null, 'holding a sign does not repeat it');
assert.equal(stream.push({ label: 'COFFEE', confidence: 0.3 }, 10), null, 'low confidence is ignored');
for (let i = 0; i < COMMIT_FRAMES; i += 1) stream.push({ label: 'COFFEE', confidence: 0.9 }, 20 + i);
assert.equal(stream.endSentence(20 + COMMIT_FRAMES), null, 'no sentence while still signing');
assert.deepEqual(stream.endSentence(20 + COMMIT_FRAMES + SENTENCE_GAP_MS), ['HELLO', 'COFFEE'], 'pause ends the sentence');
assert.equal(stream.endSentence(99999), null, 'glosses are cleared after a sentence');

assert.equal(buildPrompt({ glosses: ['ME', 'WANT', 'COFFEE'] }), 'Glosses: ME WANT COFFEE');
assert.match(buildPrompt({ glosses: ['YES'], speech: 'Want a coffee?', scene: 'a cafe counter' }), /said: "Want a coffee\?"\nThe signer is looking at: a cafe counter/);

assert.deepEqual(parseSentence('{"sentence": "I want a coffee.", "confidence": 0.93}'), { sentence: 'I want a coffee.', confidence: 0.93 });
assert.deepEqual(parseSentence('Sure!\n```json\n{"sentence":"Hello there","confidence":"0.7"}\n```'), { sentence: 'Hello there', confidence: 0.7 });
assert.deepEqual(parseSentence('I want coffee'), { sentence: 'I want coffee', confidence: 0.5 }, 'plain text falls back');
assert.equal(parseSentence(''), null);
assert.equal(parseSentence('{"sentence": "x", "confidence": 7}').confidence, 1, 'confidence is clamped');

console.log('signing.test ok');

assert.equal(matchPhrase(['WHERE', 'GATE', 'C4']), 'Where is gate C4?');
assert.equal(matchPhrase(['GATE', 'C', '4', 'WHERE']), 'Where is gate C4?', 'any order, fingerspelled C 4');
assert.equal(matchPhrase(['I', 'DEAF']), "I'm deaf.", 'I counts as ME');
assert.equal(matchPhrase(['ME', 'WANT', 'COFFEE']), null, 'other signing goes to the language model');
