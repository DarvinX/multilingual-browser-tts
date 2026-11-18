"""Use the standalone Bengali BPE and G2P for the Tiny Voca frontend."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

from .romanized_bengali import RomanizedBengali, convert_spans
from .text import BOS, EOS, PUNCTUATION, SP, SPECIALS, UNK, NothingToPronounce, Phonemized

ROOT = Path(os.environ.get("TINY_VOCA_BENGALI_FRONTEND", Path(__file__).resolve().parents[3] / "bengali-frontend"))
MODELS = ROOT / "models"
VOWELS = frozenset(("O", "E", "a", "i", "u", "e", "o", "i^", "u^", "e^", "o^"))
LOCK = threading.Lock()


@lru_cache
def symbols():
    vocab = json.loads((MODELS / "phone-vocab.json").read_text(encoding="utf-8"))
    return SPECIALS + PUNCTUATION + tuple(row["symbol"] for row in vocab if row["ipa"] is not None)


@lru_cache
def symbol_ids():
    return {symbol: i for i, symbol in enumerate(symbols())}


@lru_cache
def identity():
    return {
        name: hashlib.sha256((MODELS / name).read_bytes()).hexdigest()
        for name in ("tokenizer.model", "phonemizer.pt", "pronunciations.json", "phone-vocab.json")
    }


@lru_cache
def frontend_tools():
    with LOCK:
        # The two projects can live in adjacent checkouts or at an explicit path.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from frontend import BengaliPhonemizer, BengaliTokenizer
        return BengaliTokenizer(), BengaliPhonemizer()


@lru_cache(maxsize=1)
def romanized_bengali():
    return RomanizedBengali(frontend_tools()[1].lexicon)


def transliterate(text):
    """Convert Latin words and separated locative endings before pronunciation."""
    return convert_spans(text, lambda word: romanized_bengali().word(word))


def encode(text):
    """Return phone tokens, word-to-phone spans and the original text's BPE IDs."""
    tokenizer, phonemizer = frontend_tools()
    text_ids = tokenizer.encode(text)
    normalized = tokenizer.decode(text_ids)
    sequence, words, unknown = [BOS], [], []
    for span in phonemizer.phonemize(normalized):
        if span.kind == "punctuation":
            punctuation = span.text.translate(str.maketrans({"।": ".", "॥": ".", "-": "—"}))
            if punctuation in PUNCTUATION:
                sequence.append(punctuation)
            continue
        if len(sequence) > 1 and sequence[-1] not in PUNCTUATION:
            sequence.append(SP)
        begin = len(sequence)
        if span.kind == "word":
            sequence.extend(span.phones)
        else:
            sequence.append(UNK)
            unknown.append(span.text)
        words.append((span.text, begin, len(sequence)))
    if not words:
        raise NothingToPronounce("the text has no Bengali words")
    sequence.append(EOS)
    ids = symbol_ids()
    return Phonemized(tuple(sequence), tuple(ids[p] for p in sequence), (0,) * len(sequence), tuple(unknown)), words, text_ids
