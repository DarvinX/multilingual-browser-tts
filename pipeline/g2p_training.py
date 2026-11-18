"""Train corpus-specific BPE and a standalone recurrent G2P model."""

import argparse
from collections import Counter
from ctypes import CDLL, POINTER, byref, c_char_p, c_int
import hashlib
import json
from pathlib import Path
import re

import sentencepiece as spm

import train_phonemizer


def seed_pronunciations(language, words):
    """Generate training targets with the installed eSpeak NG pronunciation engine."""
    lib = CDLL("libespeak-ng.so.1")
    lib.espeak_Initialize.argtypes = [c_int, c_int, c_char_p, c_int]
    lib.espeak_SetVoiceByName.argtypes = [c_char_p]
    lib.espeak_TextToPhonemes.argtypes = [POINTER(c_char_p), c_int, c_int]
    lib.espeak_TextToPhonemes.restype = c_char_p
    voice = "en-us" if language == "en" else language
    if lib.espeak_Initialize(2, 0, None, 0) < 0 or lib.espeak_SetVoiceByName(voice.encode()) != 0:
        raise RuntimeError(f"eSpeak NG voice {voice} is missing")
    output = {}
    for word in words:
        source = c_char_p(word.encode())
        raw = lib.espeak_TextToPhonemes(byref(source), 1, 2 | (ord("_") << 8))
        if raw:
            ipa = re.sub(r"\([a-z-]+\)", "", raw.decode()).strip()
            phones = [p.strip() for p in ipa.split("_") if p.strip()]
            if language != "en":
                phones = [p.lstrip("ˈˌ") for p in phones]
            if phones and all(p and not any(c.isspace() or c.isdigit() for c in p) for p in phones):
                output[word] = phones
    return output


def bucket(uid):
    return int(uid[:8], 16) % 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", default=24, type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    texts = [row["text"] for row in rows if bucket(row["id"]) > 2]
    spm.SentencePieceTrainer.train(sentence_iterator=iter(texts), model_prefix=str(out / "tokenizer"),
                                   model_type="bpe", vocab_size=512, byte_fallback=True,
                                   character_coverage=1.0, normalization_rule_name="identity",
                                   hard_vocab_limit=False, num_threads=2)
    word_pattern = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE) if args.language == "en" else None
    if word_pattern:
        words = Counter(word for text in texts for word in word_pattern.findall(text.casefold()))
    else:
        from unicodedata import category
        words = Counter("".join(ch for ch in piece if category(ch)[0] in "LM") for text in texts for piece in text.split())
    seeds = seed_pronunciations(args.language, sorted(word for word in words if 0 < len(word) < 60))
    inventory = Counter(phoneme for phones in seeds.values() for phoneme in phones)
    # Keep the complete corpus phone inventory. Small phone-embedding differences
    # are measured at export, independently of the inherited acoustic architecture.
    ipa = dict((phone, phone) for phone in sorted(inventory))
    lexicon = {word: [tuple(phones)] for word, phones in seeds.items()}
    train_phonemizer.main(["--device", args.device, "--epochs", str(args.epochs), "--output", str(out)],
                          lexicon_data=(lexicon, {}, ipa))
    (out / "frontend-source.json").write_text(json.dumps({"teacher": "eSpeak NG", "language": args.language,
        "training_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "training_sentences": len(texts), "pronunciation_words": len(seeds)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
