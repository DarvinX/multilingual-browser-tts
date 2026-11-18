"""Independently transcribe generated WAVs and preserve word timings for training."""

import argparse
from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import unicodedata

import numpy as np


ASR = "Systran/faster-whisper-large-v3"
RATE = 16000


def read_rows(directory):
    with sqlite3.connect(f"{(directory / '.generation.sqlite3').resolve().as_uri()}?mode=ro", uri=True, timeout=30) as db:
        return [json.loads(row) for row, in db.execute("SELECT metadata FROM clips WHERE nsamples > 0 ORDER BY rowid")]


def normal(text):
    folded = unicodedata.normalize("NFC", text).casefold()
    return "".join(ch for ch in folded if unicodedata.category(ch)[0] in "LMN")


def distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for index, char in enumerate(reference, 1):
        next_row = [index]
        for column, predicted in enumerate(hypothesis, 1):
            next_row.append(min(row[column] + 1, next_row[-1] + 1, row[column - 1] + (char != predicted)))
        row = next_row
    return row[-1]


class Recognizer:
    def __init__(self, device):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        model = WhisperModel(ASR, device=device, compute_type="int8_float16" if device == "cuda" else "int8",
                             cpu_threads=3, local_files_only=True)
        self.pipeline = BatchedInferencePipeline(model)

    def transcribe(self, audio, language):
        starts, cursor = [], 0
        chunks, intervals = [], []
        for wave in audio:
            starts.append(cursor / RATE)
            intervals.append({"start": cursor / RATE, "end": (cursor + len(wave)) / RATE})
            # A small gap makes clip boundaries unambiguous when timestamps are rounded.
            cursor += len(wave) + RATE // 4
            chunks.extend([wave, np.zeros(RATE // 4, dtype=np.float32)])
        segments, _ = self.pipeline.transcribe(np.concatenate(chunks), language=language, beam_size=2,
                                            batch_size=2, clip_timestamps=intervals, temperature=0.0,
                                            word_timestamps=True, without_timestamps=False,
                                            condition_on_previous_text=False)
        output = [{"words": [], "hypothesis": ""} for _ in audio]
        for segment in segments:
            # Whisper can adjust the first word timestamp across a chunk edge.
            # The seek position still identifies the original input chunk.
            seek_seconds = segment.seek / self.pipeline.model.frames_per_second
            index = max(0, bisect_right(starts, seek_seconds + 0.012) - 1)
            offset = starts[index]
            row = output[index]
            row["hypothesis"] += " " + segment.text.strip()
            for word in segment.words or []:
                row["words"].append({"word": word.word.strip(), "start": max(0.0, word.start - offset),
                                     "end": min(word.end - offset, len(audio[index]) / RATE)})
        return output


def records(path):
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def audit(directory, target, language, device, limit):
    from faster_whisper.audio import decode_audio

    target.parent.mkdir(parents=True, exist_ok=True)
    available = read_rows(directory)
    if limit:
        available = random.Random(1701).sample(available, min(limit, len(available)))
    processed = {row["id"]: row.get("pcm_sha256") for row in records(target)}
    pending = [row for row in available if row["id"] not in processed or
               (processed[row["id"]] is not None and processed[row["id"]] != row["pcm_sha256"])]
    print(f"{language}: {len(pending)} recordings to audit", flush=True)
    if not pending:
        return
    model = Recognizer(device)
    if target.exists() and (data := target.read_bytes()) and not data.endswith(b"\n"):
        with target.open("r+b") as handle:
            handle.truncate(data.rfind(b"\n") + 1)
    with target.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), 32):
            bunch = pending[offset:offset + 32]
            waves = [decode_audio(directory / row["audio"], sampling_rate=RATE) for row in bunch]
            for row, result in zip(bunch, model.transcribe(waves, language), strict=True):
                reference = normal(row["text"])
                edits = distance(reference, normal(result["hypothesis"]))
                result.update(id=row["id"], edits=edits, characters=len(reference), asr_cer=edits / max(1, len(reference)),
                              pcm_sha256=row["pcm_sha256"])
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if not offset % 320:
                print(f"{language}: audited {offset + len(bunch)}/{len(pending)} pending recordings", flush=True)


def finalize(generated, audit_path, release, max_cer):
    source = json.loads((generated / "source.json").read_text())
    language, name, speaker = source["language"], source["language_name"], source["speaker"]
    all_rows = read_rows(generated)
    verified = {row["id"]: row for row in records(audit_path)}
    if set(verified) != {row["id"] for row in all_rows}:
        raise ValueError("Run the independent speech audit over all synthesized utterances first")
    accepted, total = [], 0
    for row in all_rows:
        audit_row = verified[row["id"]]
        if audit_row["asr_cer"] <= max_cer:
            accepted.append({**row, "asr_cer": audit_row["asr_cer"]})
            total += row["num_samples"]
        if total >= 20 * 3600 * 24000:
            break
    hours = total / (3600 * 24000)
    if hours < 20:
        raise ValueError(f"Only {hours:.2f} hours pass the {language} audit at CER {max_cer}")
    release.mkdir(parents=True, exist_ok=True)
    checks = [verified[row["id"]] for row in accepted]
    aggregate = sum(row["edits"] for row in checks) / sum(row["characters"] for row in checks)
    for row in accepted:
        dest = release / row["audio"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(generated / row["audio"], dest)
    for filename, rows in (("metadata.jsonl", accepted), ("asr-audit.jsonl", checks)):
        pat = release / filename
        temp = pat.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        temp.replace(pat)
    (release / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    stats = {"clips": len(accepted), "audio_hours": hours, "overall_asr_cer": aggregate,
             "maximum_clip_asr_cer": max_cer, "asr": ASR, "generated_clips": len(all_rows),
             "manifest_sha256": hashlib.sha256((release / "metadata.jsonl").read_bytes()).hexdigest()}
    (release / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    links = "\n".join(f"- [{entry['url'].rsplit('/', 1)[-1]}]({entry['url']}) · {entry['license']}"
                      for entry in source["sources"] if entry["accepted_prompts"])
    credit = "The Common Voice Sentence Collector credits Mozilla Common Voice and its contributors. "
    if any("_F.parquet" in entry["url"] for entry in source["sources"]):
        credit += "The Rasa transcripts credit [AI4Bharat and Rasa contributors](https://huggingface.co/datasets/ai4bharat/Rasa) and the indic-tts-mini-24k transcript index publisher. "
    if any("openslr.org/" in entry["url"] for entry in source["sources"]):
        credit += "The OpenSLR transcript indices credit the authors and contributors named in their resource pages. "
    if language == "ta":
        credit += "The [Tamil Wikisource contributions to Common Voice](https://github.com/common-voice/common-voice/pull/3211) credit Tamil Virtual Academy and Tamil Wikisource contributors. "
    (release / "README.md").write_text(
        f"# {name} synthetic speech · Maya 2 Native {speaker}\n\n"
        f"This release contains {len(accepted):,} {name} utterances ({hours:.3f} hours) at 24 kHz, generated with the built-in {speaker} voice. "
        "Each clip includes the synthesis prompt, source text, and provenance. Punctuation supplies the prosody hint.\n\n"
        f"## Prompt sources\n\n{credit}\n\n{links}\n\n"
        f"## Synthesis and validation\n\nSynthesis uses [Maya 2 Native](https://huggingface.co/maya-research/maya2-native/tree/{source['model_revision']}) and its "
        f"[inference engine](https://github.com/MayaResearch/maya2-native/tree/{source['engine_revision']}) under Apache 2.0, "
        "with the [Kyutai Mimi codec](https://huggingface.co/kyutai/mimi) under CC BY 4.0. "
        "The generator filters clips for normal termination, duration, silence, and clipping. "
        f"An independent [Whisper large-v3 recognizer](https://huggingface.co/{ASR}) screened the recordings at a per-clip CER of {max_cer:.0%}. "
        f"Its aggregate CER on accepted clips was {aggregate:.2%}. Recognition errors count toward CER. "
        "`asr-audit.jsonl` contains the independent hypotheses and word times.\n"
    )
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--max-cer", type=float, default=0.20)
    args = parser.parse_args()
    audit(args.generated, args.audit, args.language, args.device, args.limit)
    if args.release:
        finalize(args.generated, args.audit, args.release, args.max_cer)
