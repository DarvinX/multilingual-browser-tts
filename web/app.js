const byId = id => document.getElementById(id);
const select = byId('model');
const language = byId('language');
const load = byId('load');
const play = byId('play');
const text = byId('text');
const speed = byId('speed');
const form = byId('compose');
const player = byId('audio');
const audioToggle = byId('audio-toggle');
const seek = byId('seek');
const elapsed = byId('elapsed');
const total = byId('total');
const progress = byId('progress');
const modelStatus = byId('model-status');
const playStatus = byId('play-status');
const download = byId('download');
const examples = document.querySelectorAll('.examples button');
const bars = Array.from({ length: 34 }, () => document.createElement('i'));
byId('waveform').append(...bars);

let manifest;
let languages;
let worker;
let loaded;
let busy;
let audioUrl;
let playbackPrecision;

const mb = (bytes, digits = 2) => `${(bytes / 1e6).toFixed(digits)} MB`;

function status(element, message, error = false) {
  element.textContent = message;
  element.classList.toggle('error', error);
}

function controls() {
  const available = Boolean(manifest?.models.length);
  const ready = available && loaded === select.value;
  select.disabled = !available || Boolean(busy);
  load.disabled = !available || Boolean(busy) || ready;
  play.disabled = !ready || Boolean(busy);
  load.textContent = busy === 'load' ? 'Loading…' : ready ? 'Loaded' : 'Load model';
  play.textContent = busy === 'play' ? 'Preparing…' : 'Play';
}

function showChoice() {
  const choice = manifest.models.find(model => model.id === select.value);
  if (!choice) {
    byId('size').textContent = '';
    status(modelStatus, `${manifest.name} model is in preparation.`);
    controls();
    return;
  }
  if (choice.modelBytes) {
    byId('size').textContent = `${choice.modelBytes.toLocaleString()} bytes for this model. The text frontend and browser engine load separately.`;
    status(modelStatus, loaded === choice.id ? `${choice.label ?? choice.id.toUpperCase()} is ready.` : 'Press Load model to download this option.');
    controls();
    return;
  }
  const shared = manifest.frontendBytes + manifest.runtimeBytes;
  byId('size').textContent = worker && loaded ? `${mb(choice.bytes)} model file`
    : `${mb(choice.bytes + shared, 1)} on first load, including the speech engine`;
  status(modelStatus, loaded === choice.id ? `${choice.id.toUpperCase()} is ready.` : 'Press Load model to download this option.');
  controls();
}

function wav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const label = (offset, chars) => [...chars].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  label(0, 'RIFF');
  view.setUint32(4, buffer.byteLength - 8, true);
  label(8, 'WAVE');
  label(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  label(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) view.setInt16(44 + 2 * i, samples[i] * 32767, true);
  return new Blob([buffer], { type: 'audio/wav' });
}

function clock(seconds) {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

function updateTimeline() {
  const duration = Number.isFinite(player.duration) ? player.duration : 0;
  const position = Math.min(player.currentTime, duration);
  seek.disabled = duration === 0;
  seek.max = duration || 1;
  seek.value = position;
  elapsed.textContent = clock(position);
  total.textContent = clock(duration);
  seek.setAttribute('aria-valuetext', `${clock(position)} of ${clock(duration)}`);
  const played = duration ? position / duration : 0;
  bars.forEach((bar, index) => bar.classList.toggle('played', (index + 0.5) / bars.length <= played));
}

function updateAudioToggle() {
  const playing = !player.paused && !player.ended;
  audioToggle.dataset.playing = playing;
  audioToggle.setAttribute('aria-label', playing ? 'Pause audio' : 'Play audio');
}

function startPlayback() {
  status(playStatus, '');
  player.play().catch(() => status(playStatus, 'Press the playback button to listen.'));
}

audioToggle.addEventListener('click', () => {
  if (player.paused || player.ended) startPlayback();
  else player.pause();
});
seek.addEventListener('input', () => {
  player.currentTime = seek.valueAsNumber;
  updateTimeline();
});
for (const event of ['loadedmetadata', 'timeupdate', 'emptied']) player.addEventListener(event, updateTimeline);
for (const event of ['play', 'pause', 'ended', 'emptied']) player.addEventListener(event, updateAudioToggle);

function showAudio({ audio, sampleRate, pronounced, unknown }) {
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = URL.createObjectURL(wav(audio, sampleRate));
  player.src = audioUrl;
  download.href = audioUrl;
  download.download = `${manifest.code}-voice-${playbackPrecision}.wav`;
  byId('pronounced').textContent = pronounced;
  byId('unknown').hidden = !unknown.length;
  byId('unknown').textContent = `Unrecognized input · ${unknown.join(', ')}`;
  byId('audio-result').hidden = false;
  byId('audio-meta').textContent = playbackPrecision.toUpperCase();
  const width = Math.ceil(audio.length / bars.length);
  bars.forEach((bar, index) => {
    let sum = 0;
    const part = audio.subarray(index * width, (index + 1) * width);
    for (const sample of part) sum += sample * sample;
    const level = Math.sqrt(sum / Math.max(1, part.length));
    bar.style.setProperty('--level', `${Math.min(36, Math.max(3, level * 155))}px`);
  });
  startPlayback();
}

function startWorker() {
  worker = new Worker('worker.js', { type: 'module' });
  worker.onerror = () => {
    const target = busy === 'play' ? playStatus : modelStatus;
    status(target, 'The speech engine stopped. Try loading the model again.', true);
    worker.terminate();
    worker = undefined;
    loaded = undefined;
    busy = undefined;
    progress.hidden = true;
    controls();
  };
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') {
      progress.max = data.total;
      progress.value = Math.min(data.total, data.received);
      status(modelStatus, `Downloading model · ${mb(data.received, 1)} / ${mb(data.total, 1)}`);
      return;
    }
    if (data.type === 'starting') {
      progress.removeAttribute('value');
      status(modelStatus, 'Starting the speech engine…');
      return;
    }
    if (data.type === 'ready') {
      loaded = data.precision;
      progress.hidden = true;
    } else if (data.type === 'audio') {
      showAudio(data);
    } else if (data.type === 'error') {
      const target = data.during === 'load' ? modelStatus : playStatus;
      status(target, data.message || 'The speech engine could not finish. Try again.', true);
      progress.hidden = true;
    }
    busy = undefined;
    if (data.type === 'ready') showChoice();
    else controls();
  };
}

load.addEventListener('click', () => {
  const choice = manifest.models.find(model => model.id === select.value);
  try {
    if (!worker) startWorker();
    loaded = undefined;
    busy = 'load';
    progress.hidden = false;
    progress.removeAttribute('value');
    status(modelStatus, 'Preparing the download…');
    worker.postMessage({ type: 'load', choice, language: manifest.code });
  } catch (error) {
    busy = undefined;
    progress.hidden = true;
    status(modelStatus, error.message, true);
  }
  controls();
});

form.addEventListener('submit', event => {
  event.preventDefault();
  if (play.disabled) return;
  if (!text.value.trim()) {
    status(playStatus, 'Enter something to read aloud.', true);
    text.focus();
    return;
  }
  busy = 'play';
  playbackPrecision = loaded;
  player.pause();
  status(playStatus, 'Preparing audio…');
  controls();
  worker.postMessage({ type: 'play', text: text.value.trim(), speed: Number(speed.value),
    maxTokens: manifest.maxTokens, sampleRate: manifest.sampleRate });
});

select.addEventListener('change', showChoice);
speed.addEventListener('input', () => { byId('speed-value').textContent = `${Number(speed.value).toFixed(2)}×`; });
text.addEventListener('keydown', event => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) form.requestSubmit();
});

function count() { byId('count').textContent = `${text.value.length} / ${text.maxLength}`; }
for (const example of examples) {
  example.addEventListener('click', () => {
    text.value = example.textContent;
    count();
    text.scrollIntoView({ block: 'nearest' });
  });
}
text.addEventListener('input', count);

function switchLanguage() {
  manifest = languages.find(entry => entry.code === language.value);
  worker?.terminate();
  worker = undefined;
  loaded = undefined;
  busy = undefined;
  progress.hidden = true;
  player.pause();
  player.removeAttribute('src');
  player.load();
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = undefined;
  byId('audio-result').hidden = true;
  status(playStatus, '');

  byId('language-name').textContent = manifest.name;
  byId('script-mark').textContent = manifest.mark;
  byId('script-mark').lang = manifest.code;
  byId('voice-name').textContent = manifest.voice;
  byId('intro-copy').textContent = manifest.code === 'en'
    ? `Hear English in the ${manifest.voice} voice. Load a model and press Play.`
    : `Hear ${manifest.name} in the ${manifest.voice} voice. Type in ${manifest.name} script or Roman letters.`;
  byId('pronounced').lang = manifest.code;
  for (const [i, example] of [...examples].entries()) {
    example.textContent = manifest.examples[i];
    example.lang = i < 2 && manifest.code !== 'en' ? `${manifest.code}-Latn` : manifest.code;
  }
  text.value = manifest.examples[0];
  count();

  select.replaceChildren();
  for (const model of manifest.models) {
    select.add(new Option(`${model.label ?? model.id.toUpperCase()} · ${mb(model.modelBytes ?? model.bytes)}`, model.id));
  }
  select.value = manifest.models[0]?.id ?? '';
  showChoice();
}
language.addEventListener('change', switchLanguage);

async function initialize() {
  try {
    const response = await fetch('languages.json');
    if (!response.ok) throw new Error();
    languages = await response.json();
    for (const entry of languages) language.add(new Option(`${entry.name} · ${entry.nativeName}`, entry.code));
    language.value = 'bn';
    language.disabled = false;
    switchLanguage();
  } catch {
    status(modelStatus, 'Could not get the model options. Refresh to try again.', true);
  }
}

initialize();
