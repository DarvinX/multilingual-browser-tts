"""Approximate unfamiliar Latin spellings in Bengali script."""

from functools import lru_cache
import os
from pathlib import Path

import regex


VOWELS = {
    "a": ("আ", "া"), "ae": ("অ্যা", "্যা"), "open_o": ("অ", ""),
    "e": ("এ", "ে"), "i": ("ই", "ি"), "o": ("ও", "ো"), "u": ("উ", "ু"),
    "ai": ("আই", "াই"), "au": ("আউ", "াউ"), "oi": ("ওই", "োই"),
}
CONSONANTS = {
    "b": "ব", "bh": "ভ", "c": "ক", "ch": "চ", "chh": "ছ", "d": "ড", "dh": "ধ",
    "f": "ফ", "g": "গ", "gh": "ঘ", "h": "হ", "j": "জ", "jh": "ঝ",
    "k": "ক", "kh": "খ", "l": "ল", "m": "ম", "n": "ন", "ng": "ঙ",
    "p": "প", "ph": "ফ", "q": "ক", "r": "র", "s": "স", "sh": "শ",
    "t": "ট", "th": "থ", "v": "ভ", "w": "ওয়", "x": "ক্স", "y": "য়", "z": "জ",
}
SPELLING_VARIANTS = {
    "aa": "a", "ee": "i", "ii": "i", "oo": "u", "uu": "u",
    "ck": "k", "qu": "k", "tch": "ch",
}
LETTER_RULES = sorted(
    (key for key in VOWELS.keys() | CONSONANTS.keys() | SPELLING_VARIANTS.keys() if key.isalpha()),
    key=lambda key: (-len(key), key),
)
LETTER_PATTERN = regex.compile("|".join(LETTER_RULES))

# The pronunciation dictionary supplies ARPABET phones. The Bengali voice
# approximates several English sounds with the same Bengali vowel or consonant.
ENGLISH_SOUNDS = {
    "AA": ("a",), "AE": ("ae",), "AH": ("open_o",), "AO": ("o",), "AW": ("au",),
    "AY": ("ai",), "EH": ("e",), "ER": ("open_o", "r"), "EY": ("e",),
    "IH": ("i",), "IY": ("i",), "OW": ("o",), "OY": ("oi",), "UH": ("u",), "UW": ("u",),
    "B": ("b",), "CH": ("ch",), "D": ("d",), "DH": ("dh",), "F": ("f",),
    "G": ("g",), "HH": ("h",), "JH": ("j",), "K": ("k",), "L": ("l",), "M": ("m",),
    "N": ("n",), "NG": ("ng",), "P": ("p",), "R": ("r",), "S": ("s",),
    "SH": ("sh",), "T": ("t",), "TH": ("th",), "V": ("v",), "W": ("w",),
    "Y": ("y",), "Z": ("z",), "ZH": ("j",),
}


@lru_cache(maxsize=1)
def english_pronunciations():
    """Use a local CMU pronouncing dictionary when available."""
    path = Path(os.environ.get(
        "TINY_VOCA_ENGLISH_DICTIONARY", "/usr/share/pocketsphinx/model/en-us/cmudict-en-us.dict"
    ))
    if not path.is_file():
        return {}
    entries = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            fields = line.partition("#")[0].split()
            if not fields or not fields[0][0].isalpha():
                continue
            word, *phones = fields
            phones = [phone.rstrip("012") for phone in phones]
            if phones and all(phone in ENGLISH_SOUNDS for phone in phones):
                entries.setdefault(word.split("(", 1)[0].lower(), tuple(phones))
    return entries


def spell_sounds(sounds):
    text = []
    after_consonant = False
    for sound in sounds:
        if sound in VOWELS:
            letter, sign = VOWELS[sound]
            text.append(sign if after_consonant else letter)
            after_consonant = False
        else:
            if after_consonant:
                text.append("্")
            text.append("য" if after_consonant and sound == "y" else CONSONANTS[sound])
            after_consonant = True
    return "".join(text)


def approximate_word(word):
    phones = english_pronunciations().get(word)
    if phones:
        sounds = (sound for phone in phones for sound in ENGLISH_SOUNDS[phone])
    else:
        sounds = (SPELLING_VARIANTS.get(unit, unit) for unit in LETTER_PATTERN.findall(word))
    return spell_sounds(sounds)
