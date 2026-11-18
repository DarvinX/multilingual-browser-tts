"""Build language-specific Roman-to-native lexicons from Dakshina training data."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re
import unicodedata


SCRIPTS = {"gu": 0xA80, "hi": 0x900, "kn": 0xC80, "ml": 0xD00, "ta": 0xB80, "te": 0xC00}
CONSONANTS = {21: "k", 22: "kh", 23: "g", 24: "gh", 25: "ng", 26: "ch", 27: "chh", 28: "j", 29: "jh",
              30: "ny", 31: "t", 32: "th", 33: "d", 34: "dh", 35: "n", 36: "t", 37: "th", 38: "d", 39: "dh",
              40: "n", 41: "n", 42: "p", 43: "ph", 44: "b", 45: "bh", 46: "m", 47: "y", 48: "r", 49: "r",
              50: "l", 51: "l", 52: "zh", 53: "v", 54: "sh", 55: "sh", 56: "s", 57: "h"}
INDEPENDENT = {5: "a", 6: "aa", 7: "i", 8: "ii", 9: "u", 10: "uu", 11: "ri", 13: "e", 14: "e", 15: "ee",
               16: "ai", 17: "o", 18: "o", 19: "oo", 20: "au"}
DEPENDENT = {62: "aa", 63: "i", 64: "ii", 65: "u", 66: "uu", 67: "ri", 69: "e", 70: "e", 71: "ee",
             72: "ai", 73: "o", 74: "o", 75: "oo", 76: "au", 77: ""}


def words(text, language):
    begin = SCRIPTS[language]
    return re.findall(f"[{chr(begin)}-{chr(begin + 127)}]+", unicodedata.normalize("NFC", text))


def loose(roman):
    for long, short in (("aa", "a"), ("ee", "e"), ("ii", "i"), ("oo", "o"), ("uu", "u"), ("chh", "ch"), ("ph", "f")):
        roman = roman.replace(long, short)
    return roman


def spellings(word, language):
    base = SCRIPTS[language]
    points = [ord(ch) - base for ch in word]
    for voiced in (False, True) if language == "ta" else (False,):
        latin = []
        for i, point in enumerate(points):
            if point in CONSONANTS:
                consonant = CONSONANTS[point]
                if language == "ta" and voiced and i > 0 and points[i - 1] != 77:
                    consonant = {21: "g", 31: "d", 36: "d", 42: "b"}.get(point, consonant)
                latin.append(consonant)
                if i + 1 == len(points) or points[i + 1] not in DEPENDENT:
                    if i + 1 != len(points) or language not in ("gu", "hi"):
                        latin.append("a")
            elif point in INDEPENDENT:
                latin.append(INDEPENDENT[point])
            elif point in DEPENDENT:
                latin.append(DEPENDENT[point])
            elif point in (1, 2):
                latin.append("m" if i + 1 == len(points) else "n")
            elif language == "ml" and 122 <= point <= 127:
                latin.append(("n", "n", "r", "l", "l", "k")[point - 122])
        value = "".join(latin)
        if value:
            yield value
            yield value.replace("ii", "ee").replace("uu", "oo")


def build(source, prompt_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    for language in SCRIPTS:
        prompts = [json.loads(line)["input_text"] for line in (prompt_dir / language / "prompts.jsonl").read_text().splitlines()]
        tokens = [words(text, language) for text in prompts]
        frequency = Counter(word for sentence in tokens for word in sentence)
        bigrams = Counter((left, right) for sentence in tokens for left, right in zip(sentence, sentence[1:]))
        train = source / language / "lexicons" / f"{language}.translit.sampled.train.tsv"
        candidates = defaultdict(Counter)
        for row in train.read_text().splitlines():
            native, roman, count = row.split("\t")
            native = unicodedata.normalize("NFC", native)
            if words(native, language) == [native] and re.fullmatch(r"[a-z]+", roman):
                candidates[roman][native] += int(count)
        lexicon = {}
        for roman, options in candidates.items():
            weighted = [(word, round(math.log(1 + votes) + .33 * math.log(1 + frequency[word]), 3))
                        for word, votes in options.items()]
            lexicon[roman] = sorted(weighted, key=lambda pair: (-pair[1], pair[0]))[:3]
        aliases = defaultdict(Counter)
        for roman, entries in lexicon.items():
            for word, score in entries:
                key = loose(roman)
                aliases[key][word] = max(aliases[key][word], score)
        for word, count in frequency.items():
            if count < 2:
                continue
            for roman in spellings(word, language):
                key = loose(roman)
                score = round(.42 * math.log1p(count), 3)
                aliases[key][word] = max(aliases[key][word], score)
        relaxed = {key: counts.most_common(3) for key, counts in aliases.items() if key not in lexicon or counts.most_common(3) != lexicon[key]}
        data = {"language": language, "lexicon": lexicon,
                "relaxed": relaxed,
                "bigrams": {f"{a} {b}": count for (a, b), count in bigrams.most_common(14000) if count > 1},
                "source": "https://github.com/google-research-datasets/dakshina", "license": "CC-BY-SA-4.0"}
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
        (output / f"{language}.json.gz").write_bytes(gzip.compress(payload, mtime=0))
        print(f"{language}: {len(lexicon):,} Roman spellings", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.prompts, args.output)
