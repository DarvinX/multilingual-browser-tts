const script = { hi: 0x900, gu: 0xa80, ta: 0xb80, te: 0xc00, kn: 0xc80, ml: 0xd00 };
const vowels = {
  a: [5, 0], aa: [6, 62], i: [7, 63], ii: [8, 64], u: [9, 65], uu: [10, 66],
  e: [15, 71], ee: [15, 71], ai: [16, 72], o: [19, 75], oo: [19, 75], au: [20, 76],
};
const consonants = {
  k: 21, kh: 22, g: 23, gh: 24, ng: 25, ch: 26, chh: 27, j: 28, jh: 29, ny: 30,
  tt: 31, dd: 33, nn: 35, t: 36, th: 37, d: 38, dh: 39, n: 40, p: 42, ph: 43,
  b: 44, bh: 45, m: 46, y: 47, r: 48, l: 50, v: 53, sh: 54, s: 56, h: 57, zh: 52,
};
const variants = { q: 'k', c: 'k', f: 'ph', z: 'j', w: 'v', rr: 'r', ll: 'l' };
const tamil = {22: 21, 23: 21, 24: 21, 27: 26, 29: 28, 33: 31, 36: 36, 37: 36,
  38: 36, 39: 36, 43: 42, 44: 42, 45: 42, 54: 55 };
const units = /chh|kh|gh|ng|ch|jh|ny|tt|dd|nn|th|dh|ph|bh|sh|zh|rr|ll|aa|ii|uu|ee|oo|ai|au|[a-z]/g;
const words = /[\p{Script=Latin}]+(?:['’][\p{Script=Latin}]+)*|[\p{L}\p{M}]+|[^\p{L}\p{M}]+/gu;
const latin = /^[\p{Script=Latin}]+(?:['’][\p{Script=Latin}]+)*$/u;

function keyFor(word) {
  return word.toLowerCase().normalize('NFKD').replace(/[\p{M}'’]/gu, '')
    .replaceAll('æ', 'ae').replaceAll('ß', 'ss');
}

function loose(word) {
  const pairs = [['aa', 'a'], ['ee', 'e'], ['ii', 'i'], ['oo', 'o'], ['uu', 'u'], ['chh', 'ch'], ['ph', 'f']];
  for (const [before, after] of pairs) word = word.replaceAll(before, after);
  return word;
}

export function spellUnknown(roman, language) {
  const base = script[language];
  if (!base) return roman;
  const letters = keyFor(roman).replaceAll('x', 'ks');
  const sign = offset => offset ? String.fromCodePoint(base + offset) : '';
  let output = '';
  let pending = false;
  for (const part of letters.match(units) || []) {
    const unit = variants[part] ?? part;
    if (Object.hasOwn(vowels, unit)) {
      let offset = vowels[unit][pending ? 1 : 0];
      if (['ta', 'te', 'kn', 'ml'].includes(language) && ['e', 'o'].includes(unit)) offset--;
      output += sign(offset);
      pending = false;
      continue;
    }
    const offset = consonants[unit];
    if (!offset) continue;
    if (pending) output += sign(77);
    output += sign(language === 'ta' ? (tamil[offset] ?? offset) : offset);
    pending = true;
  }
  if (pending && ['kn', 'ml', 'ta', 'te'].includes(language)) output += sign(77);
  return output || roman;
}

export function toNative(text, language, data) {
  if (language === 'en') return text;
  const lexicon = data.lexicon;
  const chunks = [...text.normalize('NFC').matchAll(words)].map(match => match[0]);
  let previous = '';
  return chunks.map(chunk => {
    if (!latin.test(chunk)) {
      if (/^[\p{L}\p{M}]+$/u.test(chunk)) previous = chunk;
      if (/[.!?।॥]/u.test(chunk)) previous = '';
      return chunk;
    }
    const key = keyFor(chunk);
    const relaxed = loose(key);
    const alternatives = Object.hasOwn(lexicon, key) ? lexicon[key]
      : Object.hasOwn(data.relaxed, relaxed) ? data.relaxed[relaxed]
      : [[spellUnknown(chunk, language), 0]];
    const contextualScore = ([native, score]) => score + .58 * Math.log1p(data.bigrams[`${previous} ${native}`] || 0);
    const selected = alternatives.reduce((best, option) => contextualScore(option) > contextualScore(best) ? option : best);
    previous = selected[0];
    return previous;
  }).join('');
}
