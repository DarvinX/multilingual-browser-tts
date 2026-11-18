import * as ort from './runtime/ort.wasm.min.mjs';
import { finished, normalize, phonemize, romanizeInput } from './frontend.js';
import { expandWeights } from './weight-shuffle.js';
import { toNative } from './converter.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL('./runtime/', import.meta.url).href;

let voice;
let graph;
let frontend;
let g2p;
let language = 'bn';
let graphPath;
let frontendDir;
let romanData;
const predicted = new Map();
const options = { executionProviders: ['wasm'] };
const signal = (type, fields = {}) => self.postMessage({ type, ...fields });

async function unpack(file, progress) {
  const response = await fetch(new URL(file, import.meta.url));
  if (!response.ok) throw new Error(`Download failed (${response.status}). Please retry.`);
  let body = response.body;
  if (progress) {
    let bytes = 0;
    body = body.pipeThrough(new TransformStream({
      transform(chunk, stream) {
        bytes += chunk.length;
        progress(bytes);
        stream.enqueue(chunk);
      },
    }));
  }
  return new Response(body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

async function load(choice, code) {
  if (voice) {
    await voice.release();
    voice = undefined;
  }
  language = code;
  const weights = choice.weights ?? './assets/' + choice.file;
  const requestedGraph = choice.graph ?? './assets/voice.onnx.gz';
  const requestedFrontend = choice.frontend ?? './assets';
  const [model] = await Promise.all([
    unpack(weights, received => signal('progress', { received, total: choice.bytes })),
    graphPath === requestedGraph && graph || unpack(requestedGraph).then(bytes => { graph = new Uint8Array(bytes); graphPath = requestedGraph; }),
    frontendDir === requestedFrontend && frontend || unpack(requestedFrontend + '/frontend.json.gz').then(bytes => {
      frontend = JSON.parse(new TextDecoder().decode(bytes));
      frontendDir = requestedFrontend;
      predicted.clear();
      g2p?.forEach(session => session.release());
      g2p = undefined;
    }),
    ort.env.wasm.wasmBinary || unpack('./runtime/ort-wasm-simd-threaded.wasm.gz').then(bytes => { ort.env.wasm.wasmBinary = bytes; }),
  ]);
  signal('starting');
  voice = await ort.InferenceSession.create(graph, {
    ...options, externalData: [{ path: 'voice.weights', data: expandWeights(model) }],
  });
  signal('ready', { precision: choice.id });
}

function integers(values, dimensions = [1, values.length]) {
  return new ort.Tensor('int64', BigInt64Array.from(values, BigInt), dimensions);
}

async function predict(spelling) {
  if (predicted.has(spelling)) return predicted.get(spelling);
  const chars = Array.from(spelling);
  if (chars.length > 64) return [];
  if (!g2p) {
    const [enc, dec] = await Promise.all([unpack(frontendDir + '/g2p-encode.onnx.gz'), unpack(frontendDir + '/g2p-decode.onnx.gz')]);
    g2p = [await ort.InferenceSession.create(enc, options), await ort.InferenceSession.create(dec, options)];
  }
  const source = integers([1, ...chars.map(char => frontend.graphemes[char] ?? 3), 2]);
  const [encoder, decoder] = g2p;
  const { memory, keys, hidden } = await encoder.run({ source });
  let token = integers([1]);
  let state = hidden;
  const result = [];
  for (let i = 0; i < (chars.length + 2) * 2 + 4; i++) {
    const next = await decoder.run({ token, memory, keys, hidden: state });
    const id = Number(next.phone.data[0]);
    token = next.phone;
    state = next.state;
    if (id === 2) {
      if (predicted.size >= 512) predicted.clear();
      predicted.set(spelling, result);
      return result;
    }
    result.push(language === 'bn' ? frontend.phoneIds[id] : [frontend.phoneIds[id], frontend.phoneStress[id]]);
  }
  return [];
}

function scalar(value) {
  return new ort.Tensor('float32', Float32Array.of(value), [1]);
}

async function languagePhones(sentence) {
  const tokens = [frontend.special.bos];
  const stress = [0];
  const unknown = [];
  let hasWord = false;
  const marks = new Set(Object.values(frontend.marks));
  for (const match of sentence.normalize('NFC').toLowerCase().matchAll(/[\p{L}\p{M}]+(?:'[\p{L}\p{M}]+)*|[^\s]/gu)) {
    const word = match[0];
    const mark = ({ '।': '.', '॥': '.', '-': '—' })[word] ?? word;
    if (Object.hasOwn(frontend.marks, mark)) {
      tokens.push(frontend.marks[mark]);
      stress.push(0);
    } else if (/^[\p{L}\p{M}]/u.test(word)) {
      if (tokens.length > 1 && !marks.has(tokens.at(-1))) {
        tokens.push(frontend.special.space);
        stress.push(0);
      }
      const phones = Object.hasOwn(frontend.lexicon, word) ? frontend.lexicon[word] : await predict(word);
      if (!phones.length) unknown.push(word);
      for (const [phone, accent] of phones.length ? phones : [[frontend.special.unknown, 0]]) {
        tokens.push(phone);
        stress.push(accent);
      }
      hasWord ||= Boolean(phones.length);
    }
  }
  tokens.push(frontend.special.eos);
  stress.push(0);
  return { tokens, stress, unknown, hasWord };
}

async function synthesize({ text, speed, maxTokens, sampleRate }) {
  if (!voice) throw new Error('Load a voice model first.');
  let pronounced;
  if (language === 'bn') {
    pronounced = finished(normalize(romanizeInput(text, frontend.roman)));
  } else {
    let spoken = normalize(text);
    if (language !== 'en' && /\p{Script=Latin}/u.test(spoken)) {
      romanData ??= JSON.parse(new TextDecoder().decode(await unpack(`./assets/converter/${language}.json.gz`)));
      spoken = toNative(spoken, language, romanData);
    }
    pronounced = /[।॥.!?…,;:]$/.test(spoken) ? spoken : spoken + '.';
  }
  const maxPhones = Math.min(maxTokens - 12, 125);
  const unknown = new Set();
  const chunks = [];

  async function read(sentence) {
    const phones = language === 'bn' ? await phonemize(sentence, frontend, predict) : await languagePhones(sentence);
    for (const token of phones.unknown) unknown.add(token);
    if (!phones.hasWord) return;
    if (phones.tokens.length > maxPhones) {
      const clauses = sentence.split(/(?<=[,;:—])\s+/u);
      if (clauses.length > 1) {
        for (const clause of clauses) await read(clause);
        return;
      }
      const words = sentence.split(' ');
      if (words.length < 2) throw new Error('A word is too long to read aloud.');
      const middle = Math.ceil(words.length / 2);
      await read(words.slice(0, middle).join(' '));
      await read(words.slice(middle).join(' '));
      return;
    }
    const output = await voice.run({
      tokens: integers(phones.tokens), stress: integers(phones.stress ?? new Array(phones.tokens.length).fill(0)),
      voice: integers([0], [1]), speed: scalar(speed), pitch_shift: scalar(0), pitch_range: scalar(1),
    });
    chunks.push(output.audio.data);
  }

  for (const sentence of pronounced.split(/(?<=[।॥.!?…]["')\]]?)\s+/u)) await read(sentence);
  if (!chunks.length) throw new Error('Enter a word or sentence for the selected language.');
  const audio = new Float32Array(chunks.reduce((length, chunk) => length + chunk.length, 0));
  let offset = 0;
  let peak = 0;
  for (const chunk of chunks) {
    for (const sample of chunk) {
      if (!Number.isFinite(sample)) throw new Error('The voice model produced invalid audio.');
      peak = Math.max(peak, Math.abs(sample));
    }
    audio.set(chunk, offset);
    offset += chunk.length;
  }
  if (peak > 0.99) for (let i = 0; i < audio.length; i++) audio[i] *= 0.99 / peak;
  self.postMessage({ type: 'audio', audio, sampleRate, pronounced, unknown: [...unknown] }, [audio.buffer]);
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') await load(data.choice, data.language);
    if (data.type === 'play') await synthesize(data);
  } catch (error) {
    signal('error', { message: error.message, during: data.type });
  }
};
