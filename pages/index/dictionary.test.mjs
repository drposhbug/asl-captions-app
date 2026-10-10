import assert from 'node:assert/strict';
import { GLOSSES, fallbackSentence, lookup } from './dictionary.js';

const keys = Object.keys(GLOSSES);
assert.ok(keys.length >= 250, `about 250 glosses (got ${keys.length})`);
assert.deepEqual(keys, [...keys].sort(), 'glosses are alphabetical');
const POS = new Set(['noun', 'verb', 'adj', 'pron', 'wh', 'time', 'interj', 'num']);
for (const k of keys) assert.ok(POS.has(GLOSSES[k].pos) && GLOSSES[k].english, `${k} has english and a valid pos`);

assert.equal(lookup('THANK-YOU'), 'thank you');
assert.equal(lookup('ZEBRA'), 'zebra', 'unknown gloss is lowercased');

const cases = [
  [['TOMORROW', 'ME', 'GO'], 'Tomorrow I go.', 'time word first, ME as subject'],
  [['ME', 'GO', 'TOMORROW'], 'Tomorrow I go.', 'time word moved to the front'],
  [['PLEASE', 'HELP', 'ME'], 'Please help me.', 'ME after a verb is the object'],
  [['ME', 'WANT', 'COFFEE'], 'I want coffee.', 'plain statement'],
  [['MY', 'NAME', 'OSCAR'], 'My name oscar.', 'possessive kept, unknown lowercased'],
  [['BATHROOM', 'WHERE'], 'Where is the bathroom?', 'wh at the end, noun phrase'],
  [['YOUR', 'NAME', 'WHAT'], 'What is your name?', 'wh with possessive, no article'],
  [['YOU', 'NAME', 'WHAT'], 'What is your name?', 'YOU before a noun reads as your'],
  [['YOU', 'WANT', 'WHAT'], 'What do you want?', 'wh with a verb takes do'],
  [['TOMORROW', 'YOU', 'GO', 'WHERE'], 'Where do you go tomorrow?', 'time goes last in questions'],
  [['WANT', 'COFFEE', 'YOU'], 'Do you want coffee?', 'trailing YOU after a verb is a yes/no question'],
  [['UNDERSTAND', 'YOU'], 'Do you understand?', 'short yes/no question'],
  [['ME', 'HELP', 'YOU'], 'I help you.', 'YOU as object is not a question'],
  [['COFFEE', 'HOT'], 'The coffee is hot.', 'noun + adj gets the and is'],
  [['MY', 'BAG', 'LOST'], 'My bag is lost.', 'possessive noun + adj'],
  [['ME', 'TIRED'], 'I am tired.', 'I am'],
  [['YOU', 'DEAF'], 'You are deaf.', 'you are'],
  [['ME', 'NEED', 'PIZZA', 'NOW'], 'Now I need pizza.', 'now is a time word'],
  [['WHERE'], 'Where?', 'lone wh'],
  [['HELLO'], 'Hello.', 'interjection'],
  [['ME', 'DONT-UNDERSTAND'], "I don't understand.", 'compound gloss'],
];
for (const [glosses, expected, why] of cases) assert.equal(fallbackSentence(glosses), expected, why);
assert.equal(fallbackSentence([]), '', 'no glosses, no sentence');

console.log('dictionary.test ok');
