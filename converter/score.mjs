import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { toNative } from './converter.mjs';

const [assets, bengaliRoot, output] = process.argv.slice(2);
const bn = await import(new URL(`${bengaliRoot}/frontend.js`, 'file:').href);
const languageCodes = ['bn', 'gu', 'hi', 'kn', 'ml', 'ta', 'te', 'en'];
const normal = text => Array.from(text.normalize('NFC').toLowerCase().replace(/[^\p{L}\p{M}\p{N}]/gu, ''));

function distance(a, b) {
  let row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (const [i, left] of a.entries()) {
    const next = [i + 1];
    for (const [j, right] of b.entries()) next.push(Math.min(row[j] + (left !== right), row[j + 1] + 1, next[j] + 1));
    row = next;
  }
  return row.at(-1);
}

const scores = {};
for (const language of languageCodes) {
  const fixture = new URL(`./fixtures/${language}.jsonl`, import.meta.url);
  const rows = readFileSync(fixture, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  const path = language === 'bn' ? `${bengaliRoot}/assets/frontend.json.gz` : `${assets}/${language}.json.gz`;
  const data = language === 'en' ? null : JSON.parse(gunzipSync(readFileSync(path)));
  const results = rows.map(row => {
    const predicted = language === 'bn' ? bn.romanizeInput(row.roman, data.roman) : toNative(row.roman, language, data);
    const reference = normal(row.native);
    return { ...row, predicted, edits: distance(reference, normal(predicted)), characters: reference.length };
  });
  scores[language] = {
    cases: rows.length, exact: results.filter(row => row.edits === 0).length,
    character_error_rate: results.reduce((sum, row) => sum + row.edits, 0) / results.reduce((sum, row) => sum + row.characters, 0),
    unique_native: new Set(results.map(row => row.native)).size,
    unique_roman: new Set(results.map(row => row.roman)).size,
  };
  writeFileSync(`${output}/${language}.jsonl`, results.map(row => JSON.stringify(row)).join('\n') + '\n');
}
writeFileSync(`${output}/summary.json`, JSON.stringify(scores, null, 2) + '\n');
console.log(scores);
