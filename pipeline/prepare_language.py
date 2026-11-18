"""Join independent ASR word timings to learned phone sequences and acoustics."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import torch

from audit import normal, records
from language_frontend import load_frontend
from prepare_bengali import phone_durations, split_id
from pitch_viterbi import install
from tiny_voca.config import Config
from tiny_voca.journal import atomic_json, file_hash, recover
from tiny_voca.prepare import extract
from tiny_voca.text import frontend_fingerprint
from tiny_voca.train import atomic_save


def anchors(words, observed, duration):
    target = [normal(word) for word, _, _ in words]
    decoded = [normal(row["word"]) for row in observed]
    blocks = SequenceMatcher(a=target, b=decoded, autojunk=False).get_matching_blocks()
    known = {}
    for start, offset, length in blocks:
        for i in range(length):
            row = observed[offset + i]
            if 0 <= row["start"] <= row["end"] <= duration + .02:
                known[start + i] = (row["start"], row["end"])
    if len(known) < max(2, (len(words) + 1) // 2):
        raise ValueError("insufficient independent word anchors")
    positions = sorted(known)
    step = duration / len(words)
    coordinates = [-1, *positions, len(words)]
    starts = np.interp(np.arange(len(words)), coordinates, [max(0, known[positions[0]][0] - step),
                      *(known[i][0] for i in positions), duration])
    ends = np.interp(np.arange(len(words)), coordinates, [0, *(known[i][1] for i in positions), duration])
    return list(zip(starts / .04, np.maximum(starts, ends) / .04, strict=True))


def acoustic(path, config):
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    with wave.open(str(path), "rb") as source:
        signal = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if source.getframerate() != config.sample_rate:
            raise ValueError(f"Unexpected rate in {path}")
    return extract(signal, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("frontend", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    install()
    frontend = load_frontend(args.frontend)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.release / "source.json").read_text())
    config_data = torch.load(args.reference, map_location="cpu", weights_only=True)["config"]
    config = Config.from_dict({**config_data, "vocab_size": len(frontend.symbols), "speakers": [source["speaker"]]})
    atomic_json(out / "config.json", config.to_dict())
    fingerprint = frontend_fingerprint()
    identity = {"dataset": file_hash(args.release / "metadata.jsonl"), "frontend": fingerprint,
                "acoustic": config.acoustic_fingerprint, "word_anchors": file_hash(args.release / "asr-audit.jsonl")}
    annotation = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    meta = out / "annotations.json"
    if meta.exists():
        previous = json.loads(meta.read_text())
        if previous["fingerprint"] != annotation:
            raise ValueError("Annotation inputs changed; use a new output directory")
        if previous["complete"]:
            return
    atomic_json(meta, {"fingerprint": annotation, "identity": identity, "acoustic_root": str(out), "complete": False})
    audit = {row["id"]: row["words"] for row in records(args.release / "asr-audit.jsonl")}
    vowels = {frontend.symbols[index] for index in frontend.vowel_ids}
    counts = Counter()
    for split in ("train", "validation", "test"):
        accepted = out / f"{split}.jsonl"
        rejected = out / f"{split}-rejected.jsonl"
        features_path = out / split
        features_path.mkdir(exist_ok=True)
        good, bad = recover(accepted), recover(rejected)
        already = {row["id"] for row in [*good, *bad]}
        counts[split] = len(good)
        counts[f"{split}_rejected"] = len(bad)
        pending = [row for row in records(args.release / "metadata.jsonl") if split_id(row["id"]) == split and row["id"] not in already]
        with ThreadPoolExecutor(max_workers=args.workers) as workers, accepted.open("a") as accept, rejected.open("a") as reject:
            for offset in range(0, len(pending), 24):
                batch = pending[offset:offset + 24]
                pairs = [(row, features_path / f"{row['id']}.pt") for row in batch]
                tasks = [workers.submit(acoustic, dest if dest.exists() else args.release / row["audio"], config) for row, dest in pairs]
                for (row, dest), task in zip(pairs, tasks, strict=True):
                    uid, transcript = row["id"], row["text"]
                    try:
                        phone, words, bpe = frontend.encode(transcript)
                        if phone.unknown:
                            raise ValueError(f"Unknown pronunciations: {phone.unknown}")
                        spans = anchors(words, audit[uid], row["duration"])
                        bundle = task.result()
                        energy = bundle["prosody"][:, 2].numpy()
                        durations = phone_durations(phone.symbols, words, spans, energy, config, vowels)
                        if len(phone.ids) > config.max_tokens or len(energy) > config.max_frames:
                            raise ValueError("Sample exceeds model frame or token limits")
                    except ValueError as problem:
                        reject.write(json.dumps({"id": uid, "error": str(problem)}, ensure_ascii=False) + "\n")
                        counts[f"{split}_rejected"] += 1
                        continue
                    if not dest.exists():
                        atomic_save(bundle, dest)
                    item = {"id": uid, "text": transcript, "speaker": row["speaker"],
                            "text_hash": hashlib.sha256(transcript.encode()).hexdigest(), "frames": len(energy),
                            "path": str(dest.relative_to(out)), "acoustic": config.acoustic_fingerprint,
                            "frontend": fingerprint, "annotation": annotation, "tokens": phone.ids,
                            "stress": phone.stress, "durations": durations, "text_bpe_ids": bpe,
                            "audio_seconds": row["duration"]}
                    accept.write(json.dumps(item, ensure_ascii=False) + "\n")
                    counts[split] += 1
                accept.flush()
                reject.flush()
                if not offset % 240:
                    print(json.dumps(counts), flush=True)
    if any(counts[split] == 0 for split in ("train", "validation", "test")):
        raise RuntimeError(f"Missing aligned utterances for a split: {counts}")
    atomic_json(meta, {"fingerprint": annotation, "identity": identity, "acoustic_root": str(out), "complete": True, "counts": counts})


if __name__ == "__main__":
    main()
