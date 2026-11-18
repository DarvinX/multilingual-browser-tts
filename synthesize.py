"""Generate and resume a single-voice Maya 2 Native speech corpus."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import time
import wave
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from maya3 import Maya3TTS
from maya3.text import normalise_text


SAMPLE_RATE = 24_000


def json_line(record):
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def atomic_text(path, text):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def check_audio(text, pcm, info):
    if not info.get("stops") or any(not stop["reason"].startswith("aeos") for stop in info["stops"]):
        return "engine_stop_" + str(info.get("reason"))
    if pcm.ndim != 1 or pcm.dtype != np.int16 or not pcm.size:
        return "invalid_pcm"
    duration = pcm.size / SAMPLE_RATE
    if not 1.2 <= duration <= 23 or not 0.7 <= len(text.split()) / duration <= 5.5:
        return "implausible_duration"
    normalized = pcm.astype(np.float32) / 32768
    if np.sqrt(np.mean(normalized * normalized)) < 0.008:
        return "quiet"
    if np.mean(np.abs(normalized) >= 0.999) > 0.001:
        return "clipping"
    chunk = SAMPLE_RATE // 50
    blocks = normalized[:pcm.size // chunk * chunk].reshape(-1, chunk)
    active = np.sqrt(np.mean(blocks * blocks, axis=1)) > 0.0025
    if active.mean() < 0.35:
        return "excess_silence"
    return None


def synthesize(tts, item, language, voice):
    text = item["text"]
    seed = int(item["id"][:8], 16) % (2**31 - 2) + 1
    failure = "missing_audio"
    for attempt in range(2):
        use_seed = (seed + attempt) % (2**31 - 1)
        info = {}
        chunks = list(tts.stream(text, lang=language, voice=voice, normalize=False,
                                 seed=use_seed, timeout=50, info=info))
        if not chunks:
            continue
        pcm = np.concatenate(chunks)
        failure = check_audio(text, pcm, info)
        if failure is None:
            return pcm, use_seed, None
    return None, None, failure


def save_wave(path, pcm):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".wav.tmp")
    with wave.open(str(temporary), "wb") as stream:
        stream.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def normalized_prompts(path, language, limit, completed):
    source_seen = set()
    items = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            text = normalise_text(item["input_text"], language)
            if not text or text in source_seen or len(text.split()) < 4:
                continue
            source_seen.add(text)
            if item["id"] in completed:
                continue
            item["text"] = text
            items.append(item)
            if limit and len(items) >= limit:
                break
    print(f"Queued {len(items)} unique {language} prompts", flush=True)
    # Sort within small groups to keep load balanced while preserving a broad mix.
    for offset in range(0, len(items), 64):
        yield from sorted(items[offset:offset + 64], key=lambda item: len(item["text"]), reverse=True)


def digest(pcm):
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def publish(database, output, progress):
    path = output / "metadata.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for (record,) in database.execute("SELECT metadata FROM clips WHERE nsamples > 0 ORDER BY rowid"):
            handle.write(record + "\n")
    temporary.replace(path)
    atomic_text(output / "progress.json", json_line(progress))


def verify_existing(database, output):
    """Requeue rows whose recordings are missing or torn after abrupt shutdown."""
    damaged = []
    for uid, metadata in database.execute("SELECT id, metadata FROM clips WHERE nsamples > 0"):
        row = json.loads(metadata)
        try:
            with wave.open(str(output / row["audio"]), "rb") as stream:
                pcm = stream.readframes(row["num_samples"])
            if len(pcm) != row["num_samples"] * 2 or hashlib.sha256(pcm).hexdigest() != row["pcm_sha256"]:
                damaged.append((uid,))
        except (OSError, EOFError, wave.Error):
            damaged.append((uid,))
    if damaged:
        database.executemany("DELETE FROM clips WHERE id = ?", damaged)
        database.commit()
        print(f"Requeued {len(damaged)} missing or damaged audio files", flush=True)


def digest_source(source_text, path):
    checksum = hashlib.sha256(source_text.encode())
    checksum.update(path.read_bytes())
    return checksum.hexdigest()


def generate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sqlite3.connect(output / ".generation.sqlite3") as database:
            run(database, output, args)


def run(database, output, args):
    source_path = args.prompts / "source.json"
    prompt_path = args.prompts / "prompts.jsonl"
    source_json = source_path.read_text()
    source = json.loads(source_json)
    language, voice = source["language"], source["speaker"]
    identity = digest_source(source_json, prompt_path)
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=FULL")
    database.execute("CREATE TABLE IF NOT EXISTS clips (id TEXT PRIMARY KEY, nsamples INTEGER NOT NULL, metadata TEXT NOT NULL)")
    database.execute("CREATE TABLE IF NOT EXISTS settings (identity TEXT NOT NULL)")
    existing = database.execute("SELECT identity FROM settings").fetchone()
    if existing is None:
        database.execute("INSERT INTO settings VALUES (?)", (identity,))
    elif existing[0] != identity:
        raise ValueError("Prompt manifest or synthesis settings changed for this run")
    database.commit()
    atomic_text(output / "source.json", source_json)
    verify_existing(database, output)
    total, count = database.execute("SELECT coalesce(sum(nsamples), 0), count(*) FROM clips WHERE nsamples > 0").fetchone()
    if args.pilot and count:
        raise RuntimeError("Choose a fresh directory for the pilot.")
    target = math.ceil(args.hours * 3600 * SAMPLE_RATE)
    if total >= target:
        publish(database, output, {"status": "complete", "accepted_clips": count, "audio_seconds": total / SAMPLE_RATE,
                                   "target_seconds": target / SAMPLE_RATE})
        return
    completed = {row[0] for row in database.execute("SELECT id FROM clips")}
    rejects = len(completed) - count
    items = iter(normalized_prompts(prompt_path, language, args.pilot, completed))
    os.environ["MAYA3_MIMI_DIR"] = str(args.models / "mimi")
    os.environ["MAYA3_ECAPA_DIR"] = str(args.models / "ecapa")
    print(f"Loading {voice} ({language}) with {args.workers} concurrent requests", flush=True)
    tts = Maya3TTS.from_pretrained(args.models / "maya2-native", device="cuda", max_seats=args.workers)
    started = time.monotonic()
    initial_total = total
    last_status = 0.0
    failure_reasons = Counter()
    pending = {}

    def progress(status="generating"):
        elapsed = time.monotonic() - started
        speed = (total - initial_total) / SAMPLE_RATE / max(elapsed, 0.001)
        remaining = max(0, target - total) / SAMPLE_RATE
        state = {"status": status, "accepted_clips": count,
                 "rejected_clips": rejects, "audio_seconds": total / SAMPLE_RATE,
                 "target_seconds": target / SAMPLE_RATE, "wall_seconds": round(elapsed, 2),
                 "real_time_factor_x": round(speed, 3), "estimated_remaining_seconds": round(remaining / speed) if speed else None,
                 "workers": args.workers, "rejection_reasons_this_run": dict(failure_reasons)}
        atomic_text(output / "progress.json", json_line(state))
        print(json_line(state).strip(), flush=True)
        return state

    finished = False
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for item in items:
                pending[pool.submit(synthesize, tts, item, language, voice)] = item
                if len(pending) == args.workers:
                    break
            while pending and total < target and not (args.release_marker and args.release_marker.exists()):
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda job: pending[job]["id"]):
                    item = pending.pop(future)
                    pcm, seed, failure = future.result()
                    uid = item["id"]
                    if failure:
                        rejects += 1
                        failure_reasons[failure] += 1
                        record = {"id": uid, "text": item["text"], "reason": failure}
                        nsamples = 0
                    else:
                        nsamples = int(pcm.size)
                        audio = Path("audio") / uid[:2] / f"{uid}.wav"
                        record = {"id": uid, "audio": audio.as_posix(), "text": item["text"],
                                  "source_text": item["source_text"], "source_line": item["source_line"],
                                  "source": item["source"], "source_license": item["source_license"],
                                  "language": language, "speaker": voice, "prosody_hint": item["prosody_hint"],
                                  "sample_rate": SAMPLE_RATE, "num_samples": nsamples,
                                  "duration": nsamples / SAMPLE_RATE, "seed": seed, "pcm_sha256": digest(pcm)}
                        save_wave(output / audio, pcm)
                        total += nsamples
                        count += 1
                    database.execute("INSERT INTO clips VALUES (?, ?, ?)", (uid, nsamples, json_line(record).rstrip("\n")))
                    database.commit()
                    if total >= target:
                        break
                    following = next(items, None)
                    if following is not None:
                        pending[pool.submit(synthesize, tts, following, language, voice)] = following
                now = time.monotonic()
                if now - last_status >= 30:
                    progress()
                    last_status = now
            for future in pending:
                future.cancel()
        finished = True
    finally:
        tts.close()
        sealed = args.release_marker and args.release_marker.exists()
        status = "complete" if args.pilot or total >= target or sealed else "exhausted" if finished else "generating"
        publish(database, output, progress(status))
    if not args.pilot and total < target and not sealed:
        raise RuntimeError(f"Only {total / SAMPLE_RATE / 3600:.2f} hours generated before prompts ran out.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=20)
    parser.add_argument("--workers", type=int, choices=(8, 16), default=16)
    parser.add_argument("--pilot", type=int, default=0, help="Limit to this many prompts")
    parser.add_argument("--release-marker", type=Path, help="Stop once the independent audit seals the release")
    generate(parser.parse_args())
