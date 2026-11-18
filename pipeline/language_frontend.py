"""Corpus-specific tokenizer and learned phonemizer for annotation and inference."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import sentencepiece as spm
import torch

from g2p_model import EOS as PHONE_EOS, GraphemeToPhone, encode_word
from tiny_voca.text import BOS, EOS, SP, UNK, SPECIALS, PUNCTUATION, NothingToPronounce, Phonemized, register_language_frontend


WORD_PART = re.compile(r"[^\W\d_]|['’]", re.UNICODE)
VOWELS = frozenset("aɑɐɒæəɘɚɛɜɞeɤɪiɨɯoɔøɵœɶʊuʉʌyɶ")
STRESS = {"ˈ": 1, "ˌ": 2}


def segments(text):
    word = ""
    for char in unicodedata.normalize("NFC", text).casefold().replace("’", "'"):
        if (WORD_PART.fullmatch(char) and (char != "'" or word)) or unicodedata.category(char)[0] == "M":
            word += char
        else:
            if word:
                yield False, word.rstrip("'")
                word = ""
            mark = {"।": ".", "॥": ".", "-": "—"}.get(char, char)
            if mark in PUNCTUATION:
                yield True, mark
    if word:
        yield False, word.rstrip("'")


class LanguageFrontend:
    def __init__(self, directory):
        directory = Path(directory)
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(directory / "tokenizer.model"))
        self.lexicon = json.loads((directory / "pronunciations.json").read_text())
        state = torch.load(directory / "phonemizer.pt", map_location="cpu", weights_only=True)
        self.model = GraphemeToPhone(**state["config"]).eval()
        self.model.load_state_dict(state["weights"])
        self.graphemes, self.phones = state["graphemes"], state["phones"]
        self.symbols = SPECIALS + PUNCTUATION + tuple(sorted({p.lstrip("ˈˌ") for p in state["ipa"]}))
        self.ids = {label: index for index, label in enumerate(self.symbols)}
        self.vowel_ids = frozenset(self.ids[p] for p in self.symbols if p and p[0] in VOWELS and p not in SPECIALS)
        files = ("tokenizer.model", "phonemizer.pt", "pronunciations.json", "phone-vocab.json")
        assets = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}
        self.identity = {"phonemizer": "corpus-g2p", "frontend_version": 1, "assets": assets, "symbols": self.symbols}

    @lru_cache(maxsize=2048)
    def pronunciation(self, word):
        if word in self.lexicon:
            return tuple(self.lexicon[word])
        source = torch.tensor([encode_word(word, self.graphemes)])
        tokens = self.model.predict(source, max_steps=min(128, 2 * source.shape[1] + 4))[0].tolist()
        if PHONE_EOS not in tokens or tokens[0] == PHONE_EOS:
            return (UNK,)
        return tuple(self.phones[token] for token in tokens[:tokens.index(PHONE_EOS)])

    def encode(self, text):
        normalized = " ".join(unicodedata.normalize("NFC", text).split())
        bpe = self.tokenizer.encode(normalized)
        labels, stresses, words, unknown = [BOS], [0], [], []
        for is_mark, token in segments(self.tokenizer.decode(bpe)):
            if is_mark:
                labels.append(token)
                stresses.append(0)
                continue
            if len(labels) > 1 and labels[-1] not in PUNCTUATION:
                labels.append(SP)
                stresses.append(0)
            start = len(labels)
            phones = self.pronunciation(token)
            if phones == (UNK,):
                unknown.append(token)
            labels.extend(phone.lstrip("ˈˌ") for phone in phones)
            stresses.extend(STRESS.get(phone[0], 0) for phone in phones)
            words.append((token, start, len(labels)))
        if not words:
            raise NothingToPronounce("Input contains no spoken words")
        labels.append(EOS)
        stresses.append(0)
        return Phonemized(tuple(labels), tuple(self.ids[p] for p in labels), tuple(stresses), tuple(unknown)), words, bpe


def load_frontend(path):
    frontend = LanguageFrontend(path)
    register_language_frontend(frontend)
    return frontend
