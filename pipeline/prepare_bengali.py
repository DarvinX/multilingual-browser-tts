"""Create Tiny Voca supervision from Maya Bengali audio and the local frontend.

IndicConformer provides word anchors. Within a word, durations are estimates
from the ordered G2P phones, with more weight given to vowels.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from scipy.signal import resample_poly
import sentencepiece as spm
import soundfile as sf
import torch

from tiny_voca.align import ctc_viterbi, spans_from_states, pause_carrier
from tiny_voca.bengali import VOWELS, encode
from tiny_voca.config import Config
from tiny_voca.journal import atomic_json, file_hash, recover
from tiny_voca.prepare import extract
from tiny_voca.text import frontend_fingerprint, frontend_symbols, set_frontend
from pitch_viterbi import install as install_pitch_decoder


BLANK = 256


def split_id(identifier):
    n = int(identifier[:8], 16) % 100
    return "validation" if n < 2 else "test" if n == 2 else "train"


def features(task):
    audio, target, settings, cached = task
    set_frontend("bengali")
    config = Config.from_dict(settings)
    wave, rate = sf.read(audio, dtype="float32")
    if rate != config.sample_rate:
        raise ValueError(f"{audio} has unexpected rate {rate}")
    if cached is not None and cached.is_file():
        data = torch.load(cached, map_location="cpu", weights_only=True)
        if not target.exists():
            os.link(cached, target)
    else:
        data = extract(wave, config)
        pending = target.with_suffix(".tmp")
        torch.save(data, pending)
        pending.replace(target)
    return resample_poly(wave, 2, 3), data["prosody"][:, 2].numpy()


class IndicAligner:
    def __init__(self, directory, tokenizer, device):
        import onnx_asr
        import onnxruntime as ort

        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer))
        vocab = dict((int(i), p) for p, i in (row.rsplit(maxsplit=1) for row in
                     (directory / "vocab.txt").read_text(encoding="utf-8").splitlines()))
        if vocab[BLANK] != "<blk>" or any(vocab[i] != self.tokenizer.id_to_piece(i) for i in range(BLANK)):
            raise ValueError("CTC tokenizer does not match the recognizer vocabulary")
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 2
        settings.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.runtime_error = ort.capi.onnxruntime_pybind11_state.RuntimeException
        if device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("Install onnxruntime-gpu for CUDA alignment")
            providers.insert(0, ("CUDAExecutionProvider", {
                "use_tf32": "1", "cudnn_conv_algo_search": "HEURISTIC",
                "arena_extend_strategy": "kSameAsRequested",
            }))
        self.asr = onnx_asr.load_model("nemo-conformer-ctc", directory, sess_options=settings, providers=providers).asr
        if device == "cuda" and "CUDAExecutionProvider" not in self.asr._model.get_providers():
            raise RuntimeError("ONNX Runtime could not enable CUDA")
        print(json.dumps({"aligner_providers": self.asr._model.get_providers()}), flush=True)

    def emissions(self, waves):
        lengths = np.array([len(wave) for wave in waves], dtype=np.int64)
        max_batch = min(16, max(1, (16_000 * 64) // int(max(lengths))))
        if len(waves) > max_batch:
            return [row for i in range(0, len(waves), max_batch) for row in self.emissions(waves[i:i + max_batch])]
        batch = np.zeros((len(waves), max(lengths)), dtype=np.float32)
        for target, wave in zip(batch, waves, strict=True):
            target[:len(wave)] = wave
        try:
            scores, lengths = self.asr._encode(*self.asr._preprocessor(batch, lengths))
        except self.runtime_error as error:
            if len(waves) == 1 or "allocate memory" not in str(error).lower():
                raise
            mid = len(waves) // 2
            return self.emissions(waves[:mid]) + self.emissions(waves[mid:])
        return [row[:length] for row, length in zip(scores, lengths, strict=True)]

    def word_spans(self, words, emissions):
        batches = self.tokenizer.encode([word for word, _, _ in words])
        if any(not labels or self.tokenizer.unk_id() in labels for labels in batches):
            raise ValueError("CTC tokenizer has an unknown word")
        labels = [label for group in batches for label in group]
        path = ctc_viterbi(emissions, labels, blank=BLANK)
        spans = spans_from_states(path, len(labels))
        starts = np.cumsum([0] + [len(group) for group in batches])
        score = float(np.mean([emissions[a:b + 1, label].max()
                               for (a, b), label in zip(spans, labels, strict=True)]))
        return [(spans[a][0], spans[b - 1][1]) for a, b in zip(starts[:-1], starts[1:], strict=True)], score


def partition_duration(phones, frames, vowels=VOWELS):
    weights = np.array([1.45 if phone in vowels else 1.0 for phone in phones])
    remaining = frames - len(phones)
    if remaining < 0:
        raise ValueError("a word has fewer acoustic frames than phones")
    ends = np.rint(np.cumsum(weights) * (remaining / weights.sum())).astype(int)
    ends[-1] = remaining
    return (1 + np.diff(ends, prepend=0)).tolist()


def phone_durations(symbols, words, spans, energy, config, vowels=VOWELS):
    stride = 0.04 * config.sample_rate / config.hop
    quiet = energy < max(float(energy.max()) - 4.6, -6.0)
    frames = len(energy)
    phone_counts = np.array([end - begin for _, begin, end in words])
    first = min(frames - 1, round(spans[0][0] * stride))
    last = min(frames - 1, round(spans[-1][1] * stride))
    speech = np.flatnonzero(~quiet)
    if speech.size == 0:
        raise ValueError("no speech energy")
    left = min(int(speech[0]), first, frames - int(phone_counts.sum()))
    right = max(int(speech[-1]) + 1, last + 1, left + int(phone_counts.sum()))
    if left < 0 or right > frames:
        raise ValueError("not enough acoustic frames for the phones")
    durations = [0] * len(symbols)
    durations[0], durations[-1] = left, frames - right
    for i, (_, begin, end) in enumerate(words[:-1]):
        next_begin = words[i + 1][1]
        before, after = spans[i][1] * stride, spans[i + 1][0] * stride
        remaining = int(phone_counts[i + 1:].sum())
        minimum, maximum = left + int(phone_counts[i]), right - remaining
        boundary = int(np.clip(round((before + after) / 2), minimum, maximum))
        pause_start = pause_end = boundary
        if next_begin > end and maximum > minimum:
            a = max(minimum, round(before), boundary - 16)
            b = min(maximum, round(after) + 1, boundary + 16)
            quiet_frames = np.flatnonzero(quiet[a:b]) + a
            for run in np.split(quiet_frames, np.flatnonzero(np.diff(quiet_frames) != 1) + 1):
                if len(run) >= 4 and len(run) > pause_end - pause_start:
                    pause_start, pause_end = int(run[0]), int(run[-1]) + 1
            durations[pause_carrier(symbols, range(end, next_begin))] = pause_end - pause_start
        durations[begin:end] = partition_duration(symbols[begin:end], pause_start - left, vowels)
        left = pause_end
    _, begin, end = words[-1]
    durations[begin:end] = partition_duration(symbols[begin:end], right - left, vowels)
    if sum(durations) != frames or min(durations) < 0:
        raise ValueError("durations do not cover the feature frames")
    return durations


def prepare(args):
    set_frontend("bengali")
    torch.set_num_threads(1)
    source = args.dataset / "metadata.jsonl"
    out = args.output.resolve()
    c = Config(speakers=("Jahnavi",), vocab_size=len(frontend_symbols()))
    fingerprint = frontend_fingerprint()
    identity = {
        "source": file_hash(source), "frontend": fingerprint, "acoustic": c.acoustic_fingerprint,
        "recognizer": file_hash(args.recognizer / "model.onnx"), "ctc_tokenizer": file_hash(args.ctc_tokenizer),
        "duration_method": 1, "limit_per_split": args.limit_per_split,
    }
    if args.acoustic_cache:
        cached_metadata = json.loads((args.acoustic_cache / "annotations.json").read_text())
        if cached_metadata["identity"]["acoustic"] != c.acoustic_fingerprint:
            raise ValueError("cached acoustic features use different settings")
    annotation = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    metadata_path = out / "annotations.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old["fingerprint"] != annotation:
            raise ValueError("preparation inputs changed, use a new output directory")
        if old["complete"]:
            print(json.dumps(old), flush=True)
            return
    metadata = {"fingerprint": annotation, "acoustic_root": str(out), "identity": identity, "complete": False}
    atomic_json(metadata_path, metadata)
    atomic_json(out / "config.json", c.to_dict())
    splits = ("train", "validation", "test")
    rows = {split: [] for split in splits}
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        split = split_id(row["id"])
        if not args.limit_per_split or len(rows[split]) < args.limit_per_split:
            rows[split].append(row)
    paths = {key: out / (key + ".jsonl") for key in (*splits, "rejected")}
    seen = {key: {r["id"] for r in recover(path)} for key, path in paths.items()}

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=install_pitch_decoder) as pool:
        aligner = IndicAligner(args.recognizer, args.ctc_tokenizer, args.device)
        started = time.monotonic()
        last_log = started
        rejects = paths["rejected"].open("a", encoding="utf-8")
        with rejects:
            for split in splits:
                output = out / split
                output.mkdir(exist_ok=True)
                pending = sorted((r for r in rows[split] if r["id"] not in seen[split]
                                  and r["id"] not in seen["rejected"]), key=lambda row: row["duration"])
                with paths[split].open("a", encoding="utf-8") as stream:
                    for offset in range(0, len(pending), args.batch_size):
                        batch = pending[offset:offset + args.batch_size]
                        targets = [output / (r["id"] + ".pt") for r in batch]
                        prepared = list(pool.map(features, [(args.dataset / r["audio"], target, c.to_dict(),
                                                            args.acoustic_cache / split / (r["id"] + ".pt")
                                                            if args.acoustic_cache else None)
                                                           for r, target in zip(batch, targets, strict=True)]))
                        emissions = aligner.emissions([wave for wave, energy in prepared])
                        for row, target, (_, energy), scores in zip(batch, targets, prepared, emissions, strict=True):
                            identifier, transcript = row["id"], row["text"]
                            try:
                                phone, word_spans, text_ids = encode(transcript)
                                if phone.unknown:
                                    raise ValueError(f"unsupported Bengali word {phone.unknown}")
                                spans, score = aligner.word_spans(word_spans, scores)
                                durations = phone_durations(phone.symbols, word_spans, spans, energy, c)
                            except ValueError as problem:
                                rejects.write(json.dumps({"id": identifier, "split": split, "error": str(problem)},
                                                         ensure_ascii=False) + "\n")
                                seen["rejected"].add(identifier)
                            else:
                                result = {
                                    "id": identifier, "text": transcript, "speaker": row["speaker"],
                                    "text_hash": hashlib.sha256(transcript.encode()).hexdigest(), "frames": len(energy),
                                    "path": str(target.relative_to(out)), "acoustic": c.acoustic_fingerprint,
                                    "frontend": fingerprint, "annotation": annotation, "tokens": phone.ids,
                                    "stress": phone.stress, "durations": durations, "text_bpe_ids": text_ids,
                                    "ctc_mean_label_logp": score, "audio_seconds": row["duration"],
                                }
                                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                                seen[split].add(identifier)
                        stream.flush()
                        rejects.flush()
                        if time.monotonic() - last_log > 15 or offset == 0:
                            print(json.dumps({**{k: len(v) for k, v in seen.items()},
                                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
                            last_log = time.monotonic()
    counts = {k: len(v) for k, v in seen.items()}
    if any(counts[split] == 0 for split in splits):
        raise ValueError(f"empty split: {counts}")
    metadata.update(complete=True, counts=counts)
    atomic_json(metadata_path, metadata)
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--ctc-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--acoustic-cache", type=Path, help="Reuse compatible waveform and acoustic feature tensors")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with (arguments.output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(arguments)
