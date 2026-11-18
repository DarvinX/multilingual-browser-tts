"""Validate a speech release, build Hugging Face audio shards, and upload it."""

import argparse
import hashlib
import io
import json
import shutil
import wave
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Features, Value
from huggingface_hub import HfApi


FEATURES = Features({
    "id": Value("string"), "audio": Audio(sampling_rate=24000), "text": Value("string"),
    "source_text": Value("string"), "source_line": Value("int64"), "source": Value("string"),
    "source_license": Value("string"),
    "language": Value("string"), "speaker": Value("string"), "prosody_hint": Value("string"),
    "sample_rate": Value("int32"), "num_samples": Value("int32"), "duration": Value("float64"),
    "seed": Value("int64"), "pcm_sha256": Value("string"), "asr_cer": Value("float64"),
})
SPLITS = ("train", "validation", "test")


def split_for(uid):
    bucket = int(uid[:8], 16) % 100
    return "validation" if bucket < 2 else "test" if bucket == 2 else "train"


def package(release, output, language, min_hours):
    source = json.loads((release / "source.json").read_text())
    rows = [json.loads(line) for line in (release / "metadata.jsonl").read_text().splitlines()]
    hours = sum(row["num_samples"] for row in rows) / (24000 * 3600)
    if hours < min_hours or {row["language"] for row in rows} != {language}:
        raise ValueError(f"Unexpected language or hours in {release}: {hours:.4f} hours")
    if any(len({row[key] for row in rows}) != len(rows) for key in ("id", "audio", "text")):
        raise ValueError("Release contains duplicate IDs, paths, or transcripts")
    if len({row["speaker"] for row in rows}) != 1:
        raise ValueError("Expected one speaker for the release")

    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    stats = {}
    schema = FEATURES.arrow_schema
    for split in SPLITS:
        subset = [row for row in rows if split_for(row["id"]) == split]
        stats[split] = {"clips": len(subset), "hours": sum(r["duration"] for r in subset) / 3600}
        for offset in range(0, len(subset), 1000):
            dest = data / f"{split}-{offset // 1000:05d}.parquet"
            temporary = dest.with_suffix(".tmp")
            with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
                for i in range(offset, min(offset + 1000, len(subset)), 32):
                    batch = []
                    for item in subset[i:min(i + 32, offset + 1000)]:
                        audio = (release / item["audio"]).read_bytes()
                        with wave.open(io.BytesIO(audio)) as stream:
                            params = (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes())
                            pcm = stream.readframes(params[-1])
                        if params != (1, 2, 24000, item["num_samples"]) or hashlib.sha256(pcm).hexdigest() != item["pcm_sha256"]:
                            raise ValueError(f"Audio failed integrity validation: {item['id']}")
                        row = {key: item.get(key) for key in FEATURES}
                        row["source"] = item.get("source") or source.get("source")
                        row["source_license"] = item.get("source_license") or source.get("source_license")
                        row["audio"] = {"bytes": audio, "path": Path(item["audio"]).name}
                        batch.append(row)
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            temporary.replace(dest)
            print(f"Packed {dest.name}", flush=True)

    share_alike = any(row.get("source_license") == "CC-BY-SA-4.0" for row in rows)
    license_id = "cc-by-sa-4.0" if share_alike else "cc-by-4.0"
    license_name = "CC BY-SA 4.0" if share_alike else "CC BY 4.0"
    license_url = "https://creativecommons.org/licenses/by-sa/4.0/" if share_alike else "https://creativecommons.org/licenses/by/4.0/"
    card = ["---", f"license: {license_id}", "language:", f"- {language}", "task_categories:",
            "- text-to-speech", "tags:", "- synthetic-speech", "- maya2-native", "configs:",
            "- config_name: default", "  data_files:"]
    for split in SPLITS:
        card.extend([f"  - split: {split}", f'    path: "data/{split}-*.parquet"'])
    card.extend(["---", "", (release / "README.md").read_text().strip(), "", "## Dataset splits", "",
                 "| Split | Clips | Audio hours |", "| --- | ---: | ---: |"])
    for split, info in stats.items():
        card.append(f"| {split} | {info['clips']:,} | {info['hours']:.3f} |")
    card.extend(["", "The split is assigned by a fixed hash bucket of the prompt ID. Audio is losslessly stored in the Parquet shards.",
                 "", "## Release license", "", f"The synthetic recordings and original release annotations are released under [{license_name}]({license_url}). Source prompts retain their original licenses. Credit the upstream sources and this dataset when redistributing it.", ""])
    (output / "README.md").write_text("\n".join(card))
    (output / "splits.json").write_text(json.dumps(stats, indent=2) + "\n")
    for filename in ("summary.json", "source.json", "asr-audit.jsonl", "asr-audit-summary.json"):
        if (release / filename).exists():
            shutil.copy2(release / filename, output / filename)
    print(json.dumps({"clips": len(rows), "hours": hours, "splits": stats}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--min-hours", type=float, default=20)
    parser.add_argument("--repo")
    args = parser.parse_args()
    package(args.release, args.output, args.language, args.min_hours)
    if args.repo:
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
        result = api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=args.output,
                                   commit_message="Publish validated single-voice synthetic speech and dataset card")
        files = api.list_repo_files(args.repo, repo_type="dataset", revision=result.oid)
        expected = {p.relative_to(args.output).as_posix() for p in args.output.rglob("*") if p.is_file()}
        if not expected <= set(files):
            raise RuntimeError("The dataset commit is missing release files")
        print(result.commit_url, flush=True)


if __name__ == "__main__":
    main()
