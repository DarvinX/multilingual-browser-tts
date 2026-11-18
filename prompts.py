"""Fetch pinned text-only sources and create traceable Maya prompt manifests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gzip
import json
from pathlib import Path
import random
import re
import unicodedata
from urllib.request import urlopen

import pyarrow.parquet as pq


COMMON_VOICE = "605c25e48f9fb87cb25ca794cb39bf91d3741f29"
RASA_SUBSET = "8fe16c93456540d567f4ae952adb18ab173912b3"
OPEN_SLR = {"gu": 78, "kn": 79, "ml": 63, "te": 66}
MAYA_REVISION = "87a83efdd6551e038246b6b37836a6264713eec3"
ENGINE_REVISION = "c6b376868e268551d065c4206193e96257bef4a1"
LANGUAGES = {
    "bn": ("Bengali", "Jahnavi", 0x0980), "gu": ("Gujarati", "Jahnavi", 0x0A80),
    "hi": ("Hindi", "Ishwari", 0x0900), "kn": ("Kannada", "Jahnavi", 0x0C80),
    "ml": ("Malayalam", "Jahnavi", 0x0D00), "ta": ("Tamil", "Kayal", 0x0B80),
    "te": ("Telugu", "Sreekar", 0x0C00), "en": ("English", "Nivedita", None),
}
PUNCTUATION = frozenset(' .,!?;:।॥’\'-–—“”')


def cv(language, filename):
    path = f"{COMMON_VOICE}/server/data/{language}/{filename}"
    return (f"https://raw.githubusercontent.com/common-voice/common-voice/{path}",
            f"https://github.com/common-voice/common-voice/blob/{path}", "CC0-1.0")


def sources(language):
    if language != "gu":
        yield cv(language, "sentence-collector.txt")
    if language == "ta":
        yield cv(language, "wikisource-ta.txt")
    if language not in {"en", "bn"}:
        root = "https://huggingface.co/datasets/sulabhkatiyar/indic-tts-mini-24k"
        for voice in ("F", "M"):
            path = f"{RASA_SUBSET}/index/parts/{language}_{voice}.parquet"
            yield f"{root}/resolve/{path}", f"{root}/blob/{path}", "CC-BY-4.0"
    if language in OPEN_SLR:
        index = OPEN_SLR[language]
        for voice in ("female", "male"):
            file = f"line_index_{voice}.tsv"
            yield f"https://www.openslr.org/resources/{index}/{file}", f"https://www.openslr.org/{index}/#{file}", "CC-BY-SA-4.0"


def fetch(url, output):
    if not output.is_file():
        with urlopen(url, timeout=30) as response:
            payload = response.read()
        temporary = output.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(output)
    return output


def read_text(path):
    if path.suffix == ".parquet":
        table = pq.read_table(path, columns=["text", "source_dataset"])
        if set(table["source_dataset"].to_pylist()) != {"ai4bharat/Rasa"}:
            raise ValueError(f"Unexpected attribution in {path}")
        return table["text"].to_pylist()
    if path.suffix == ".tsv":
        return [line.partition("\t")[2] for line in path.read_text(encoding="utf-8-sig").splitlines()]
    return path.read_text(encoding="utf-8-sig").splitlines()


def is_acceptable(text, script):
    def allowed(ch):
        if ch in PUNCTUATION:
            return True
        if script is None:
            return ch.isascii() and ch.isalpha()
        return script <= ord(ch) < script + 128 and unicodedata.category(ch)[0] in "LM"

    return 20 <= len(text) <= 220 and 4 <= len(text.split()) <= 35 and all(map(allowed, text))


def prepare(language, directory, reserve=None):
    name, voice, script = LANGUAGES[language]
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    bibliography = []
    for url, citation, license_id in sources(language):
        path = directory / url.rsplit("/", 1)[-1]
        fetch(url, path)
        count = 0
        for number, raw in enumerate(read_text(path), start=1):
            text = " ".join(unicodedata.normalize("NFC", raw).split())
            if not is_acceptable(text, script):
                continue
            key = "".join(ch for ch in text.casefold() if ch not in PUNCTUATION)
            if key in records:
                continue
            end = "।" if script is not None and language in {"hi", "bn"} else "."
            prompt = text if text.endswith(("?", "!", ".", "।", "॥")) else text + end
            records[key] = {
                "id": hashlib.sha256(f"{language}\0{key}".encode()).hexdigest()[:20],
                "input_text": prompt, "source_text": raw, "source_line": number,
                "source": citation, "source_license": license_id,
                "prosody_hint": "question" if text.endswith("?") else "exclamation" if text.endswith("!") else "statement",
            }
            count += 1
        bibliography.append({"url": citation, "download": url, "license": license_id, "accepted_prompts": count,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    prompts = list(records.values())
    rng = random.Random(19402)
    rng.shuffle(prompts)
    # Give conversational questions and exclamations sufficient coverage even when a large source is sampled.
    prompts.sort(key=lambda r: (r["prosody_hint"] == "statement") and len(r["input_text"].split()) < 6)
    prompts = prompts[:32000]
    if reserve and language in OPEN_SLR and len(prompts) < 32000:
        original_count = len(prompts)
        path = reserve / language / "native_script_wikipedia" / f"{language}.wiki-filt.train.text.shuf.txt.gz"
        citation = "https://github.com/google-research-datasets/dakshina#native-script-wikipedia"
        with gzip.open(path, "rt", encoding="utf-8") as text_source:
            for number, raw in enumerate(text_source, start=1):
                text = " ".join(unicodedata.normalize("NFC", raw).split())
                if not is_acceptable(text, script) or len(text.split()) < 9:
                    continue
                key = "".join(ch for ch in text.casefold() if ch not in PUNCTUATION)
                if key in records:
                    continue
                uid = hashlib.sha256(f"{language}\0{key}".encode()).hexdigest()[:20]
                item = {"id": uid, "input_text": text if text.endswith(("?", "!", ".", "।")) else text + ".",
                        "source_text": raw.rstrip("\n"), "source_line": number, "source": citation,
                        "source_license": "CC-BY-SA-4.0", "prosody_hint": "question" if text.endswith("?") else
                        "exclamation" if text.endswith("!") else "statement"}
                records[key] = item
                prompts.append(item)
                if len(prompts) >= 32000:
                    break
        bibliography.append({"url": citation, "license": "CC-BY-SA-4.0", "accepted_prompts": len(prompts) - original_count,
                             "filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with (directory / "prompts.jsonl").open("w", encoding="utf-8") as stream:
        for record in prompts:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    source = {"language": language, "language_name": name, "speaker": voice, "model": "maya-research/maya2-native",
              "model_revision": MAYA_REVISION, "engine_revision": ENGINE_REVISION,
              "candidate_count": len(prompts), "sources": bibliography}
    (directory / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    print(f"{name}: {len(prompts):,} candidate prompts, Maya voice {voice}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("languages", nargs="+", choices=LANGUAGES)
    parser.add_argument("--reserve-dir", type=Path)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda language: prepare(language, args.output / language, args.reserve_dir), args.languages))
