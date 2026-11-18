const BENGALI_LETTER = '[\\u0980-\\u09ff]';
const ROMAN_SPAN = new RegExp(
  `(?<![\\p{L}\\p{M}\\p{N}_])(?:([a-z]{2,}|${BENGALI_LETTER}+)(?:[ \\t]+|[ \\t]*[-‐‑][ \\t]*)[ae]|([a-z]+))(?![\\p{L}\\p{M}\\p{N}_])`,
  'giu',
);
const SPANS = /([\u0980-\u09ff]+(?:[-'][\u0980-\u09ff]+)*)|[\p{L}\p{M}\p{N}]+|[^\s]/gu;
const DIGITS = /[০-৯]/u;

function lookup(map, key) {
  return Object.hasOwn(map, key) ? map[key] : undefined;
}

function relaxed(word) {
  const vowel = { aa: 'a', ee: 'i', ii: 'i', oo: 'u', uu: 'u' };
  return word.replace(/aa|ee|ii|oo|uu/g, match => vowel[match])
    .replaceAll('chh', 'ch').replaceAll('cch', 'chch').replaceAll('sh', 's')
    .replaceAll('ph', 'f').replaceAll('v', 'bh');
}

function attachLocative(word, exceptions) {
  const last = word.at(-1);
  if (lookup(exceptions, word)) return exceptions[word];
  if ('আা'.includes(last)) return word + 'য়';
  if ('অইঈউঊঋএঐওঔিীুূৃেৈোৌ'.includes(last)) return word + 'তে';
  if ('ংঃ'.includes(last)) return word + 'য়ে';
  return word + 'ে';
}

export function romanizeInput(text, data) {
  const letterPattern = new RegExp(data.rules.join('|'), 'g');
  function word(spelling) {
    const key = spelling.toLowerCase();
    const known = lookup(data.months, key) || lookup(data.exact, key)
      || lookup(data.relaxed, relaxed(key)) || lookup(data.english, key);
    if (known) return known;
    let result = '';
    let afterConsonant = false;
    for (const unit of key.match(letterPattern) || []) {
      const sound = lookup(data.variants, unit) || unit;
      const vowel = lookup(data.vowels, sound);
      if (vowel) {
        result += vowel[afterConsonant ? 1 : 0];
      } else {
        if (afterConsonant) result += '্';
        result += afterConsonant && sound === 'y' ? 'য' : data.consonants[sound];
      }
      afterConsonant = !vowel;
    }
    return result || spelling;
  }
  return text.replace(ROMAN_SPAN, (_, base, latin) => base
    ? attachLocative(/^[a-z]/i.test(base) ? word(base) : base, data.locative) : word(latin));
}

export function normalize(text) {
  return text.normalize('NFKC').replace(/[\u200c\u200d]/g, '').replaceAll('’', "'")
    .replace(/[‐‑]/g, '-').replace(/\s+/g, ' ').trim();
}

export function finished(text) {
  return /[।॥.!?…,;:]["'”’)\]»]*$/.test(text) ? text : text + '।';
}

export async function phonemize(sentence, data, predict) {
  const tokens = [data.special.bos];
  const unknown = [];
  let hasWord = false;

  function mark(spelling) {
    const normalized = ({ '।': '.', '॥': '.', '-': '—' })[spelling] || spelling;
    if (Object.hasOwn(data.marks, normalized)) tokens.push(data.marks[normalized]);
  }

  async function word(spelling, bengali) {
    const phones = lookup(data.lexicon, spelling) || (bengali ? await predict(spelling) : []);
    if (tokens.at(-1) === data.special.bos || !Object.values(data.marks).includes(tokens.at(-1))) {
      if (tokens.length > 1) tokens.push(data.special.space);
    }
    if (phones.length) {
      tokens.push(...phones);
      hasWord = true;
    } else {
      unknown.push(spelling);
      tokens.push(data.special.unknown);
    }
  }

  for (const [spelling, bengali] of sentence.matchAll(SPANS)) {
    if (bengali && !DIGITS.test(spelling)) {
      if (lookup(data.lexicon, spelling)) await word(spelling, true);
      else {
        for (const part of spelling.split(/([-'])/)) {
          if (part === '-' || part === "'") mark(part);
          else await word(part, true);
        }
      }
    } else if (/^\p{P}$/u.test(spelling)) mark(spelling);
    else await word(spelling, false);
  }
  tokens.push(data.special.eos);
  return { tokens, hasWord, unknown };
}
