"""Convert Romanized Bengali input using a lexicon and spelling rules."""

import regex

from .bengali_letter_rules import approximate_word


BENGALI_LETTER = r"[\p{Script=Bengali}&&[\p{L}\p{M}]]"
BENGALI_WORD = regex.compile(rf"{BENGALI_LETTER}+", regex.V1)

# Conventional Bengali spellings for the English names.
MONTHS = {
    "january": "জানুয়ারি", "february": "ফেব্রুয়ারি", "march": "মার্চ",
    "april": "এপ্রিল", "may": "মে", "june": "জুন", "july": "জুলাই",
    "august": "আগস্ট", "september": "সেপ্টেম্বর", "october": "অক্টোবর",
    "november": "নভেম্বর", "december": "ডিসেম্বর",
}
LOCATIVE_EXCEPTIONS = {"মে": "মে মাসে", "জুলাই": "জুলাইয়ে"}
WORD_CHARACTER = r"[\p{L}\p{M}\p{N}_]"
ROMANIZED_SPAN = regex.compile(
    rf"(?<!{WORD_CHARACTER})(?:(?P<base>[a-zA-Z]{{2,}}|{BENGALI_LETTER}+)"
    rf"(?:[ \t]+|[ \t]*[-‐‑][ \t]*)[aAeE]|(?P<word>[a-zA-Z]+))(?!{WORD_CHARACTER})",
    regex.V1,
)


def with_locative(word):
    """Attach the Bengali locative ending to a Bengali spelling."""
    if word in LOCATIVE_EXCEPTIONS:
        return LOCATIVE_EXCEPTIONS[word]
    if word[-1] in "আা":
        return word + "য়"
    if word[-1] in "অইঈউঊঋএঐওঔিীুূৃেৈোৌ":
        return word + "তে"
    if word[-1] in "ংঃ":
        return word + "য়ে"
    return word + "ে"


def convert_spans(text, romanized_word):
    def replace(match):
        base = match["base"]
        if base:
            word = romanized_word(base) if base.isascii() else base
            return with_locative(word)
        return romanized_word(match["word"])

    return ROMANIZED_SPAN.sub(replace, text)


# Keep the lexicon's phone labels separate from their usual keyboard spellings.
ROMAN_PHONE = {
    "D": "d", "Dh": "dh", "E": "e", "N": "ng", "O": "o", "T": "t", "Th": "th",
    "c": "ch", "ch": "chh", "e^": "e", "i^": "i", "o^": "o", "u^": "u",
}

# Frequent spellings that are ambiguous or diverge from the pronunciation.
ALIASES = {
    "আমি": ("ami",),
    "বাংলা": ("bangla",),
    "বাঙালি": ("bangali",),
    "বেঙ্গলি": ("bengali",),
    "ভালোবাসি": ("bhalobasi", "bhalobashi", "valobasi", "valobashi"),
    "ভালো": ("bhalo", "valo"),
    "তোমার": ("tomar",),
    "তোমাকে": ("tomake",),
    "আছো": ("acho", "achho"),
    "আছি": ("achi", "achhi"),
    "আছে": ("ache", "achhe"),
    "কোথায়": ("kothay",),
    "যাচ্ছ": ("jaccho", "jachcho", "jachchho"),
    "যাচ্ছি": ("jacchi", "jachchi", "jachchhi"),
    "যাচ্ছেন": ("jacchen", "jachchen", "jachchhen"),
    "যাবো": ("jabo",),
    "যেতে": ("jete",),
    "হবে": ("hobe",),
    "কথা": ("kotha", "katha"),
    "চলো": ("cholo", "chalo"),
    "সকাল": ("sokal", "shokal", "sakal", "shakal"),
    "আবহাওয়া": ("abohawa", "abohowa"),
    "খাবো": ("khabo",),
    "খাওয়া": ("khawa", "khaoa"),
    "ধন্যবাদ": ("dhonnobad", "dhanyabad"),
    "সঙ্গে": ("songe", "shonge", "sange"),
    "সাথে": ("sathe", "shathe"),
    "অনেক": ("onek", "anek"),
    "নতুন": ("notun", "natun"),
    "গল্প": ("golpo", "golpa"),
    "অফিস": ("office",),
}
DOUBLED_VOWEL = regex.compile(r"(aa|ee|ii|oo|uu)")
VOWEL_VARIANTS = {"aa": "a", "ee": "i", "ii": "i", "oo": "u", "uu": "u"}


def relaxed_spelling(word):
    """Fold common keyboard spelling variants for a second dictionary lookup."""
    word = DOUBLED_VOWEL.sub(lambda match: VOWEL_VARIANTS[match[0]], word)
    return word.replace("chh", "ch").replace("cch", "chch").replace("sh", "s").replace("ph", "f").replace("v", "bh")


class RomanizedBengali:
    def __init__(self, pronunciations):
        self.exact = {}
        self.relaxed = {}
        alternate_vowels = {}
        for word, phones in pronunciations.items():
            if not BENGALI_WORD.fullmatch(word):
                continue
            roman = "".join(ROMAN_PHONE.get(phone, phone) for phone in phones)
            self.exact.setdefault(roman, word)
            self.relaxed.setdefault(relaxed_spelling(roman), word)
            if "O" in phones:
                alternate = "".join("a" if phone == "O" else ROMAN_PHONE.get(phone, phone) for phone in phones)
                alternate_vowels.setdefault(relaxed_spelling(alternate), word)
        for roman, word in alternate_vowels.items():
            self.relaxed.setdefault(roman, word)
        for word, aliases in ALIASES.items():
            for alias in aliases:
                self.exact[alias] = word
                self.relaxed[relaxed_spelling(alias)] = word

    def word(self, spelling):
        roman = spelling.lower()
        if roman in MONTHS:
            return MONTHS[roman]
        return self.exact.get(roman) or self.relaxed.get(relaxed_spelling(roman)) or approximate_word(roman)
