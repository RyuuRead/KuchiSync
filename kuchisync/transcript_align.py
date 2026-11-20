# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass
import math
import re

import numpy as np

from kuchisync import audio_dsp

PHRASE_SPLIT_RE = re.compile(r"[.,?!;:\r\n]+")
WORD_SPLIT_RE = re.compile(r"[A-Za-z\u00C0-\u024F']+")

PENULTIMATE_WEIGHT = 1.35
PHRASE_FINAL_WEIGHT = 1.25
DEFAULT_WEIGHT = 0.90

VOCAL_LOW_HZ = 300.0
VOCAL_HIGH_HZ = 3000.0
ENVELOPE_WINDOW_MS = 40.0
MAX_GAP_MERGE_MS = 120.0
MIN_ISLAND_MS = 30.0

_ACCENT_TRANS = str.maketrans({
    "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a", "å": "a",
    "é": "e", "è": "e", "ê": "e", "ë": "e",
    "í": "i", "ì": "i", "î": "i", "ï": "i",
    "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o",
    "ú": "u", "ù": "u", "û": "u", "ü": "u",
})

_INDONESIAN_VOWELS = frozenset(("a", "i", "u", "e", "o"))
_INDONESIAN_DIPHTHONGS = frozenset(("ai", "au", "oi", "ei"))
_INDONESIAN_DIGRAPHS = ("ng", "ny", "kh", "sy")

_ENGLISH_VOWEL_CHARS = frozenset(("a", "e", "i", "o", "u"))

_ARPABET_TO_AIUEO: dict[str, tuple[str, ...]] = {
    "AA": ("A",),
    "AE": ("A",),
    "AH": ("A",),
    "IY": ("I",),
    "IH": ("I",),
    "UW": ("U",),
    "UH": ("U",),
    "EH": ("E",),
    "EY": ("E",),
    "ER": ("E",),
    "AO": ("O",),
    "OW": ("O",),
    "AY": ("A", "I"),
    "AW": ("A", "U"),
    "OY": ("O", "I"),
}

# Common and irregular English words stored as ARPAbet vowel syllable tuples.
_ENGLISH_ARPABET_LEXICON: dict[str, tuple[str, ...]] = {
    "a": ("AH",),
    "about": ("AH", "AW"),
    "above": ("AH", "AH"),
    "again": ("AH", "EH"),
    "against": ("AH", "EH"),
    "all": ("AO",),
    "allow": ("AH", "AW"),
    "already": ("AO", "EH", "IY"),
    "always": ("AO", "EY"),
    "am": ("AE",),
    "an": ("AE",),
    "and": ("AE",),
    "another": ("AH", "AH", "ER"),
    "answer": ("AE", "ER"),
    "any": ("EH", "IY"),
    "are": ("AA",),
    "aren't": ("AA",),
    "around": ("ER", "AW"),
    "as": ("AE",),
    "at": ("AE",),
    "away": ("AH", "EY"),
    "baby": ("EY", "IY"),
    "be": ("IY",),
    "beautiful": ("UW", "IH", "AH"),
    "because": ("IH", "AO"),
    "become": ("IH", "AH"),
    "been": ("IH",),
    "before": ("IH", "AO"),
    "begin": ("IH", "IH"),
    "believe": ("IH", "IY"),
    "better": ("EH", "ER"),
    "between": ("IH", "IY"),
    "bone": ("OW",),
    "book": ("UH",),
    "both": ("OW",),
    "bought": ("AO",),
    "break": ("EY",),
    "bring": ("IH",),
    "brought": ("AO",),
    "build": ("IH",),
    "built": ("IH",),
    "busy": ("IH", "IY"),
    "but": ("AH",),
    "buy": ("AY",),
    "by": ("AY",),
    "call": ("AO",),
    "came": ("EY",),
    "can": ("AE",),
    "can't": ("AE",),
    "cannot": ("AE", "AA"),
    "catch": ("AE",),
    "caught": ("AO",),
    "change": ("EY",),
    "child": ("AY",),
    "children": ("IH", "AH"),
    "choose": ("UW",),
    "city": ("IH", "IY"),
    "close": ("OW",),
    "clothes": ("OW",),
    "color": ("AH", "ER"),
    "come": ("AH",),
    "coming": ("AH", "IH"),
    "could": ("UH",),
    "couldn't": ("UH", "AH"),
    "country": ("AH", "IY"),
    "course": ("AO",),
    "create": ("IY", "EY"),
    "cut": ("AH",),
    "daughter": ("AO", "ER"),
    "day": ("EY",),
    "dead": ("EH",),
    "death": ("EH",),
    "decide": ("IH", "AY"),
    "did": ("IH",),
    "didn't": ("IH", "AH"),
    "die": ("AY",),
    "different": ("IH", "ER", "AH"),
    "do": ("UW",),
    "does": ("AH",),
    "doesn't": ("AH", "AH"),
    "don't": ("OW",),
    "done": ("AH",),
    "door": ("AO",),
    "down": ("AW",),
    "draw": ("AO",),
    "drink": ("IH",),
    "drive": ("AY",),
    "each": ("IY",),
    "early": ("ER", "IY"),
    "earth": ("ER",),
    "easy": ("IY", "IY"),
    "eat": ("IY",),
    "eight": ("EY",),
    "either": ("IY", "ER"),
    "else": ("EH",),
    "end": ("EH",),
    "english": ("IH", "IH"),
    "enough": ("IH", "AH"),
    "even": ("IY", "AH"),
    "ever": ("EH", "ER"),
    "every": ("EH", "IY"),
    "everything": ("EH", "IY", "IH"),
    "example": ("IH", "AE", "AH"),
    "eye": ("AY",),
    "face": ("EY",),
    "fall": ("AO",),
    "family": ("AE", "AH", "IY"),
    "far": ("AA",),
    "father": ("AA", "ER"),
    "feel": ("IY",),
    "feet": ("IY",),
    "few": ("UW",),
    "fight": ("AY",),
    "find": ("AY",),
    "fine": ("AY",),
    "first": ("ER",),
    "five": ("AY",),
    "fly": ("AY",),
    "follow": ("AA", "OW"),
    "food": ("UW",),
    "foot": ("UH",),
    "for": ("AO",),
    "forever": ("ER", "EH", "ER"),
    "forget": ("ER", "EH"),
    "found": ("AW",),
    "four": ("AO",),
    "friend": ("EH",),
    "from": ("AH",),
    "full": ("UH",),
    "future": ("UW", "ER"),
    "game": ("EY",),
    "gave": ("EY",),
    "get": ("EH",),
    "girl": ("ER",),
    "give": ("IH",),
    "go": ("OW",),
    "goes": ("OW",),
    "going": ("OW", "IH"),
    "gone": ("AO",),
    "good": ("UH",),
    "got": ("AA",),
    "great": ("EY",),
    "group": ("UW",),
    "grow": ("OW",),
    "guess": ("EH",),
    "had": ("AE",),
    "hadn't": ("AE", "AH"),
    "half": ("AE",),
    "hand": ("AE",),
    "happy": ("AE", "IY"),
    "hard": ("AA",),
    "has": ("AE",),
    "hasn't": ("AE", "AH"),
    "have": ("AE",),
    "haven't": ("AE", "AH"),
    "he": ("IY",),
    "he's": ("IY",),
    "head": ("EH",),
    "hear": ("IH",),
    "heard": ("ER",),
    "heart": ("AA",),
    "heavy": ("EH", "IY"),
    "hello": ("EH", "OW"),
    "her": ("ER",),
    "here": ("IH",),
    "high": ("AY",),
    "him": ("IH",),
    "his": ("IH",),
    "hold": ("OW",),
    "home": ("OW",),
    "hope": ("OW",),
    "hour": ("AW", "ER"),
    "house": ("AW",),
    "how": ("AW",),
    "however": ("AW", "EH", "ER"),
    "human": ("UW", "AH"),
    "i": ("AY",),
    "i'd": ("AY",),
    "i'll": ("AY",),
    "i'm": ("AY",),
    "i've": ("AY",),
    "idea": ("AY", "IY", "AH"),
    "if": ("IH",),
    "important": ("IH", "AO", "AH"),
    "in": ("IH",),
    "inside": ("IH", "AY"),
    "instead": ("IH", "EH"),
    "into": ("IH", "UW"),
    "is": ("IH",),
    "island": ("AY", "AH"),
    "isle": ("AY",),
    "isn't": ("IH", "AH"),
    "it": ("IH",),
    "it's": ("IH",),
    "its": ("IH",),
    "just": ("AH",),
    "keep": ("IY",),
    "key": ("IY",),
    "knew": ("UW",),
    "knife": ("AY",),
    "know": ("OW",),
    "known": ("OW",),
    "language": ("AE", "IH"),
    "large": ("AA",),
    "last": ("AE",),
    "late": ("EY",),
    "laugh": ("AE",),
    "learn": ("ER",),
    "leave": ("IY",),
    "left": ("EH",),
    "less": ("EH",),
    "let": ("EH",),
    "letter": ("EH", "ER"),
    "lie": ("AY",),
    "life": ("AY",),
    "light": ("AY",),
    "like": ("AY",),
    "line": ("AY",),
    "listen": ("IH", "AH"),
    "little": ("IH", "AH"),
    "live": ("IH",),
    "long": ("AO",),
    "look": ("UH",),
    "lose": ("UW",),
    "lost": ("AO",),
    "love": ("AH",),
    "made": ("EY",),
    "make": ("EY",),
    "man": ("AE",),
    "many": ("EH", "IY"),
    "may": ("EY",),
    "me": ("IY",),
    "mean": ("IY",),
    "meant": ("EH",),
    "meet": ("IY",),
    "men": ("EH",),
    "might": ("AY",),
    "mind": ("AY",),
    "mine": ("AY",),
    "minute": ("IH", "AH"),
    "money": ("AH", "IY"),
    "month": ("AH",),
    "moon": ("UW",),
    "more": ("AO",),
    "morning": ("AO", "IH"),
    "most": ("OW",),
    "mother": ("AH", "ER"),
    "move": ("UW",),
    "much": ("AH",),
    "music": ("UW", "IH"),
    "must": ("AH",),
    "my": ("AY",),
    "name": ("EY",),
    "near": ("IH",),
    "need": ("IY",),
    "neither": ("IY", "ER"),
    "never": ("EH", "ER"),
    "new": ("UW",),
    "night": ("AY",),
    "nine": ("AY",),
    "no": ("OW",),
    "not": ("AA",),
    "nothing": ("AH", "IH"),
    "now": ("AW",),
    "number": ("AH", "ER"),
    "ocean": ("OW", "AH"),
    "of": ("AH",),
    "off": ("AO",),
    "often": ("AO", "AH"),
    "old": ("OW",),
    "on": ("AA",),
    "once": ("AH",),
    "one": ("AH",),
    "only": ("OW", "IY"),
    "open": ("OW", "AH"),
    "or": ("AO",),
    "order": ("AO", "ER"),
    "other": ("AH", "ER"),
    "our": ("AW",),
    "out": ("AW",),
    "outside": ("AW", "AY"),
    "over": ("OW", "ER"),
    "own": ("OW",),
    "paper": ("EY", "ER"),
    "part": ("AA",),
    "party": ("AA", "IY"),
    "past": ("AE",),
    "pay": ("EY",),
    "people": ("IY", "AH"),
    "person": ("ER", "AH"),
    "picture": ("IH", "ER"),
    "piece": ("IY",),
    "pie": ("AY",),
    "place": ("EY",),
    "play": ("EY",),
    "please": ("IY",),
    "point": ("OY",),
    "poor": ("UH",),
    "possible": ("AA", "AH", "AH"),
    "power": ("AW", "ER"),
    "pretty": ("IH", "IY"),
    "probably": ("AA", "AH", "IY"),
    "problem": ("AA", "AH"),
    "pull": ("UH",),
    "push": ("UH",),
    "put": ("UH",),
    "question": ("EH", "AH"),
    "quiet": ("AY", "AH"),
    "quite": ("AY",),
    "read": ("IY",),
    "ready": ("EH", "IY"),
    "real": ("IY",),
    "really": ("IY", "IY"),
    "reason": ("IY", "AH"),
    "remember": ("IH", "EH", "ER"),
    "right": ("AY",),
    "room": ("UW",),
    "round": ("AW",),
    "run": ("AH",),
    "said": ("EH",),
    "same": ("EY",),
    "saw": ("AO",),
    "say": ("EY",),
    "says": ("EH",),
    "school": ("UW",),
    "science": ("AY", "AH"),
    "sea": ("IY",),
    "second": ("EH", "AH"),
    "see": ("IY",),
    "seem": ("IY",),
    "seen": ("IY",),
    "self": ("EH",),
    "send": ("EH",),
    "sense": ("EH",),
    "sent": ("EH",),
    "seven": ("EH", "AH"),
    "several": ("EH", "ER", "AH"),
    "shall": ("AE",),
    "she": ("IY",),
    "she's": ("IY",),
    "should": ("UH",),
    "shouldn't": ("UH", "AH"),
    "show": ("OW",),
    "side": ("AY",),
    "sign": ("AY",),
    "simple": ("IH", "AH"),
    "since": ("IH",),
    "sing": ("IH",),
    "sister": ("IH", "ER"),
    "sit": ("IH",),
    "six": ("IH",),
    "sleep": ("IY",),
    "small": ("AO",),
    "smile": ("AY",),
    "snow": ("OW",),
    "so": ("OW",),
    "some": ("AH",),
    "someone": ("AH", "AH"),
    "something": ("AH", "IH"),
    "sometimes": ("AH", "AY"),
    "son": ("AH",),
    "song": ("AO",),
    "soon": ("UW",),
    "sorry": ("AA", "IY"),
    "sound": ("AW",),
    "speak": ("IY",),
    "special": ("EH", "AH"),
    "stand": ("AE",),
    "start": ("AA",),
    "state": ("EY",),
    "stay": ("EY",),
    "still": ("IH",),
    "stop": ("AA",),
    "story": ("AO", "IY"),
    "street": ("IY",),
    "strong": ("AO",),
    "study": ("AH", "IY"),
    "such": ("AH",),
    "sure": ("UH",),
    "system": ("IH", "AH"),
    "table": ("EY", "AH"),
    "take": ("EY",),
    "talk": ("AO",),
    "teach": ("IY",),
    "tell": ("EH",),
    "ten": ("EH",),
    "than": ("AE",),
    "thank": ("AE",),
    "that": ("AE",),
    "that's": ("AE",),
    "the": ("AH",),
    "their": ("EH",),
    "them": ("EH",),
    "then": ("EH",),
    "there": ("EH",),
    "there's": ("EH",),
    "these": ("IY",),
    "they": ("EY",),
    "they're": ("EH",),
    "thing": ("IH",),
    "think": ("IH",),
    "third": ("ER",),
    "this": ("IH",),
    "those": ("OW",),
    "though": ("OW",),
    "thought": ("AO",),
    "three": ("IY",),
    "through": ("UW",),
    "throw": ("OW",),
    "thus": ("AH",),
    "tie": ("AY",),
    "time": ("AY",),
    "to": ("UW",),
    "today": ("AH", "EY"),
    "together": ("AH", "EH", "ER"),
    "told": ("OW",),
    "tomorrow": ("AH", "AA", "OW"),
    "tonight": ("AH", "AY"),
    "too": ("UW",),
    "took": ("UH",),
    "touch": ("AH",),
    "town": ("AW",),
    "tree": ("IY",),
    "true": ("UW",),
    "try": ("AY",),
    "turn": ("ER",),
    "twelve": ("EH",),
    "twenty": ("EH", "IY"),
    "twice": ("AY",),
    "two": ("UW",),
    "under": ("AH", "ER"),
    "understand": ("AH", "ER", "AE"),
    "until": ("AH", "IH"),
    "up": ("AH",),
    "upon": ("AH", "AA"),
    "us": ("AH",),
    "use": ("UW",),
    "used": ("UW",),
    "usually": ("UW", "AH", "IY"),
    "very": ("EH", "IY"),
    "voice": ("OY",),
    "wait": ("EY",),
    "walk": ("AO",),
    "wall": ("AO",),
    "want": ("AA",),
    "warm": ("AO",),
    "was": ("AH",),
    "wasn't": ("AH", "AH"),
    "wash": ("AA",),
    "watch": ("AA",),
    "water": ("AO", "ER"),
    "way": ("EY",),
    "we": ("IY",),
    "we're": ("IH",),
    "wear": ("EH",),
    "week": ("IY",),
    "weight": ("EY",),
    "well": ("EH",),
    "went": ("EH",),
    "were": ("ER",),
    "weren't": ("ER",),
    "what": ("AH",),
    "what's": ("AH",),
    "when": ("EH",),
    "where": ("EH",),
    "which": ("IH",),
    "while": ("AY",),
    "white": ("AY",),
    "who": ("UW",),
    "whole": ("OW",),
    "whose": ("UW",),
    "why": ("AY",),
    "wide": ("AY",),
    "wife": ("AY",),
    "will": ("IH",),
    "wind": ("IH",),
    "window": ("IH", "OW"),
    "wish": ("IH",),
    "with": ("IH",),
    "without": ("IH", "AW"),
    "woman": ("UH", "AH"),
    "women": ("IH", "IH"),
    "won": ("AH",),
    "won't": ("OW",),
    "wonder": ("AH", "ER"),
    "word": ("ER",),
    "wore": ("AO",),
    "work": ("ER",),
    "world": ("ER",),
    "would": ("UH",),
    "wouldn't": ("UH", "AH"),
    "write": ("AY",),
    "written": ("IH", "AH"),
    "wrong": ("AO",),
    "wrote": ("OW",),
    "yeah": ("AE",),
    "year": ("IH",),
    "yellow": ("EH", "OW"),
    "yes": ("EH",),
    "yet": ("EH",),
    "you": ("UW",),
    "you'll": ("UW",),
    "you're": ("UH",),
    "you've": ("UW",),
    "young": ("AH",),
    "your": ("AO",),
}

_LONG_VOWEL_TO_AIUEO: dict[str, tuple[str, ...]] = {
    "a": ("E",),
    "e": ("I",),
    "i": ("A", "I"),
    "o": ("O",),
    "u": ("U",),
    "y": ("A", "I"),
}

_SHORT_VOWEL_TO_AIUEO: dict[str, tuple[str, ...]] = {
    "a": ("A",),
    "e": ("E",),
    "i": ("I",),
    "o": ("O",),
    "u": ("A",),
    "y": ("I",),
}

_ENGLISH_DIGRAPHS: dict[str, tuple[str, ...]] = {
    "ee": ("I",),
    "ea": ("I",),
    "oo": ("U",),
    "ai": ("E",),
    "ay": ("E",),
    "oi": ("O", "I"),
    "oy": ("O", "I"),
    "au": ("O",),
    "aw": ("O",),
    "oa": ("O",),
    "ie": ("I",),
    "ei": ("E",),
    "eu": ("U",),
    "ew": ("U",),
    "oe": ("O",),
    "ue": ("U",),
    "ui": ("U",),
    "ou": ("A", "U"),
}

_OW_O_STUBS = frozenset((
    "know", "show", "snow", "grow", "throw", "blow", "flow", "slow", "low",
    "glow", "row", "crow", "stow", "mow", "sow", "tow", "bow", "own",
    "bowl", "blown", "grown", "known", "shown", "thrown", "flown",
))


class TranscriptError(ValueError):
    """Raised when a transcript contains no extractable vowel targets."""


@dataclass
class PhonemeTarget:
    """Single vowel articulation target produced by G2P."""

    vowel: str
    weight: float
    phrase_index: int
    same_as_prev_in_word: bool


@dataclass
class AlignedSyllable:
    """Acoustic-aligned syllable span mapped to Blender timeline frames."""

    vowel: str
    start_frame: float
    end_frame: float
    peak_rms_db: float
    valley_dip_before: bool


def _normalize_language(language: str) -> str:
    """Normalize language tag to INDONESIAN or ENGLISH."""
    norm = str(language).strip().upper()
    if norm in ("ENGLISH", "EN", "ENG"):
        return "ENGLISH"
    return "INDONESIAN"


def _clean_word_letters(raw: str) -> str:
    """Lowercase and normalize accents while keeping ASCII letters and apostrophes."""
    lowered = raw.lower().translate(_ACCENT_TRANS)
    return re.sub(r"[^a-z']", "", lowered)


def _tokenize_indonesian_units(word: str) -> list[str]:
    """Split an Indonesian word into graphemic consonant/digraph/vowel units."""
    units: list[str] = []
    idx = 0
    n = len(word)
    while idx < n:
        if idx + 1 < n and word[idx : idx + 2] in _INDONESIAN_DIGRAPHS:
            units.append(word[idx : idx + 2])
            idx += 2
        else:
            units.append(word[idx])
            idx += 1
    return units


def _indonesian_word_syllables(word: str) -> list[tuple[str, ...]]:
    """Split an Indonesian word into vowel/diphthong tuples per syllable."""
    clean = re.sub(r"[^a-z]", "", word.lower().translate(_ACCENT_TRANS))
    if not clean:
        return []
    units = _tokenize_indonesian_units(clean)
    n = len(units)
    syllables: list[tuple[str, ...]] = []
    idx = 0
    while idx < n:
        ch = units[idx]
        if ch in _INDONESIAN_VOWELS:
            if idx + 1 < n and units[idx + 1] in _INDONESIAN_VOWELS:
                pair = ch + units[idx + 1]
                at_end = idx + 2 == n
                followed_by_cv = (
                    idx + 3 < n
                    and units[idx + 2] not in _INDONESIAN_VOWELS
                    and units[idx + 3] in _INDONESIAN_VOWELS
                )
                if pair in _INDONESIAN_DIPHTHONGS and (at_end or followed_by_cv):
                    syllables.append((ch.upper(), units[idx + 1].upper()))
                    idx += 2
                    continue
            syllables.append((ch.upper(),))
        idx += 1
    return syllables


def _is_english_vowel_at(word: str, idx: int) -> bool:
    """Return True if word[idx] acts as a vowel (including non-initial/non-glide y)."""
    ch = word[idx]
    if ch in _ENGLISH_VOWEL_CHARS:
        if ch == "u" and idx > 0 and word[idx - 1] == "q":
            return False
        return True
    if ch == "y":
        if idx == 0:
            return False
        if idx + 1 < len(word) and word[idx + 1] in _ENGLISH_VOWEL_CHARS:
            return False
        return True
    return False


def _has_english_vowel(word: str) -> bool:
    """Return True if word contains at least one orthographic vowel."""
    return any(_is_english_vowel_at(word, i) for i in range(len(word)))


def _is_single_magic_e_stem(stem: str) -> bool:
    """Return True if stem ends with single vowel + single consonant (e.g. bak, tim)."""
    if len(stem) < 3:
        return False
    c = stem[-1]
    v = stem[-2]
    if c in _ENGLISH_VOWEL_CHARS or c in ("w", "x", "y"):
        return False
    if v not in _ENGLISH_VOWEL_CHARS and v != "y":
        return False
    if stem[-3] in _ENGLISH_VOWEL_CHARS or stem[-3] == "y":
        return False
    vowel_groups = 0
    in_v = False
    for i in range(len(stem)):
        is_v = _is_english_vowel_at(stem, i)
        if is_v and not in_v:
            vowel_groups += 1
        in_v = is_v
    return vowel_groups == 1


def _english_core_vowels(word: str) -> list[tuple[str, ...]]:
    """Rule-based English orthographic vowel extractor for unseen words."""
    if not word or not _has_english_vowel(word):
        return []

    w = word
    # Drop silent final e when an earlier vowel exists (keep open monosyllables like be/he/ye).
    silent_e_magic = False
    if w.endswith("e") and len(w) >= 3:
        if w.endswith("gue") or w.endswith("que"):
            if _has_english_vowel(w[:-2]):
                w = w[:-2]
        elif w[-2] not in _ENGLISH_VOWEL_CHARS and _has_english_vowel(w[:-1]):
            w = w[:-1]
            silent_e_magic = _is_single_magic_e_stem(w)

    syllables: list[tuple[str, ...]] = []
    idx = 0
    n = len(w)
    while idx < n:
        ch = w[idx]
        if ch == "q" and idx + 1 < n and w[idx + 1] == "u":
            idx += 2
            continue
        if idx == 0 and ch == "g" and n >= 3 and w[1] == "u" and w[2] in ("e", "i"):
            idx += 2
            continue

        if not _is_english_vowel_at(w, idx):
            idx += 1
            continue

        # Trigraph / quadgraph patterns before standard digraphs.
        if w.startswith("eigh", idx):
            syllables.append(("E",))
            idx += 4
            continue
        if w.startswith("igh", idx):
            syllables.append(("A", "I"))
            idx += 3
            continue
        if w.startswith("augh", idx):
            syllables.append(("O",))
            idx += 4
            continue
        if w.startswith("ough", idx):
            syllables.append(("O",))
            idx += 4
            continue

        pair = w[idx : idx + 2]
        if len(pair) == 2 and (pair in _ENGLISH_DIGRAPHS or pair in ("ow", "ey")):
            if pair == "ow":
                at_end = idx + 2 == n
                has_earlier = len(syllables) > 0
                if w in _OW_O_STUBS or (at_end and has_earlier):
                    syllables.append(("O",))
                else:
                    syllables.append(("A", "U"))
                idx += 2
                continue
            if pair == "ey":
                at_end = idx + 2 == n
                if at_end and len(syllables) > 0:
                    syllables.append(("I",))
                else:
                    syllables.append(("E",))
                idx += 2
                continue
            if pair == "ie" and idx + 2 == n and len(syllables) == 0:
                syllables.append(("A", "I"))
                idx += 2
                continue
            syllables.append(_ENGLISH_DIGRAPHS[pair])
            idx += 2
            continue

        # Single vowel (check magic-e lengthening on last vowel, -ind/-ild, -al, -y).
        is_last_vowel = not any(_is_english_vowel_at(w, j) for j in range(idx + 1, n))
        if silent_e_magic and is_last_vowel:
            syllables.append(_LONG_VOWEL_TO_AIUEO.get(ch, ("E",)))
            idx += 1
            continue

        if ch == "y":
            if idx + 1 == n:
                syllables.append(("I",) if len(syllables) > 0 else ("A", "I"))
            else:
                syllables.append(("I",))
            idx += 1
            continue

        if ch == "a" and idx + 2 <= n and w[idx : idx + 2] == "al":
            after_al = w[idx + 2 : idx + 3]
            if after_al in ("l", "k", "t", ""):
                syllables.append(("O",))
                idx += 1
                continue

        if ch == "i" and idx + 2 < n and w[idx + 1 : idx + 3] in ("nd", "ld") and idx + 3 == n:
            syllables.append(("A", "I"))
            idx += 1
            continue

        if ch == "u":
            # Open single-u syllable (flu, menu, super, music) vs closed short u (cup, jump).
            if idx + 1 == n:
                syllables.append(("U",))
            elif (
                idx + 2 < n
                and w[idx + 1] not in _ENGLISH_VOWEL_CHARS
                and _is_english_vowel_at(w, idx + 2)
            ):
                syllables.append(("U",))
            else:
                syllables.append(("A",))
            idx += 1
            continue

        if ch == "e" and idx + 1 == n and len(syllables) == 0:
            syllables.append(("I",))
            idx += 1
            continue

        syllables.append(_SHORT_VOWEL_TO_AIUEO.get(ch, ("E",)))
        idx += 1

    return syllables


def _english_word_syllables(word: str) -> list[tuple[str, ...]]:
    """Extract English vowel/diphthong tuples via lexicon + morphological rules."""
    clean = _clean_word_letters(word).strip("'")
    if not clean:
        return []

    if clean in _ENGLISH_ARPABET_LEXICON:
        return [_ARPABET_TO_AIUEO[p] for p in _ENGLISH_ARPABET_LEXICON[clean]]

    # Handle possessive 's and common contractions cleanly before inflectional rules.
    if clean.endswith("'s") and len(clean) > 2:
        stem = clean[:-2]
        base = _english_word_syllables(stem)
        if stem.endswith(("s", "x", "z", "ch", "sh", "ce", "ge")):
            return base + [("I",)]
        return base

    for suffix in ("'ll", "'ve", "'re", "'d", "'m"):
        if clean.endswith(suffix) and len(clean) > len(suffix):
            return _english_word_syllables(clean[: -len(suffix)])

    letters = re.sub(r"[^a-z]", "", clean)
    if not letters:
        return []
    if letters in _ENGLISH_ARPABET_LEXICON:
        return [_ARPABET_TO_AIUEO[p] for p in _ENGLISH_ARPABET_LEXICON[letters]]

    # Handle -ied / -ed.
    if letters.endswith("ied") and len(letters) > 3:
        prefix = letters[:-3]
        if not _has_english_vowel(prefix):
            return [("A", "I")]
        return _english_word_syllables(prefix + "y")

    if letters.endswith("ed") and len(letters) >= 4:
        stem = letters[:-2]
        if _has_english_vowel(stem):
            voiced_ed = stem.endswith(("t", "d"))
            if _is_single_magic_e_stem(stem):
                base = _english_word_syllables(stem + "e")
            elif len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in _ENGLISH_VOWEL_CHARS:
                base = _english_word_syllables(stem[:-1])
            else:
                base = _english_word_syllables(stem)
            return base + ([("I",)] if voiced_ed else [])

    # Handle -ies / -oes / -es.
    if letters.endswith("ies") and len(letters) > 3:
        prefix = letters[:-3]
        if not _has_english_vowel(prefix):
            return [("A", "I")]
        return _english_word_syllables(prefix + "y")

    if letters.endswith("oes") and len(letters) > 3 and _has_english_vowel(letters[:-2]):
        return _english_word_syllables(letters[:-2])

    if letters.endswith("es") and len(letters) >= 4:
        stem = letters[:-2]
        if _has_english_vowel(stem):
            if stem.endswith(("ch", "sh", "ss", "x", "zz")):
                return _english_word_syllables(stem) + [("I",)]
            if letters[:-1].endswith(("ce", "ge", "se", "ze")):
                return _english_word_syllables(letters[:-1]) + [("I",)]
            return _english_word_syllables(letters[:-1])

    # Handle syllabic -le after a consonant (e.g. table, little, apple).
    if (
        letters.endswith("le")
        and len(letters) >= 4
        and letters[-3] not in _ENGLISH_VOWEL_CHARS
        and letters[-3] != "y"
        and _has_english_vowel(letters[:-2])
    ):
        stem = letters[:-2]
        if _is_single_magic_e_stem(stem):
            long_v = _LONG_VOWEL_TO_AIUEO.get(stem[-2], ("E",))
            prefix = _english_core_vowels(stem[:-2]) if _has_english_vowel(stem[:-2]) else []
            return prefix + [long_v, ("A",)]
        return _english_core_vowels(stem) + [("A",)]

    # Handle trailing plural/3sg -s when stem exists in lexicon or ends cleanly.
    if letters.endswith("s") and not letters.endswith(("ss", "us", "is")) and len(letters) >= 3:
        stem = letters[:-1]
        if stem in _ENGLISH_ARPABET_LEXICON:
            return [_ARPABET_TO_AIUEO[p] for p in _ENGLISH_ARPABET_LEXICON[stem]]

    return _english_core_vowels(letters)


def _build_phrase_targets(
    word_syllables_list: list[list[tuple[str, ...]]],
    phrase_index: int,
) -> list[PhonemeTarget]:
    """Assign stress weights and intra-word repetition markers for a phrase."""
    targets: list[PhonemeTarget] = []
    n_words = len(word_syllables_list)
    for w_idx, syls in enumerate(word_syllables_list):
        is_last_word = w_idx == n_words - 1
        n_syl = len(syls)
        prev_vowel: str | None = None
        for s_idx, vowel_tuple in enumerate(syls):
            if n_syl >= 2 and s_idx == n_syl - 2:
                weight = PENULTIMATE_WEIGHT
            elif is_last_word and s_idx == n_syl - 1:
                weight = PHRASE_FINAL_WEIGHT
            else:
                weight = DEFAULT_WEIGHT

            for vowel in vowel_tuple:
                same_as_prev = prev_vowel is not None and vowel == prev_vowel
                prev_vowel = vowel
                targets.append(
                    PhonemeTarget(
                        vowel=vowel,
                        weight=weight,
                        phrase_index=phrase_index,
                        same_as_prev_in_word=same_as_prev,
                    )
                )
    return targets


def extract_vowels(text: str, language: str = "INDONESIAN") -> list[list[PhonemeTarget]]:
    """Extract phrase-grouped A-I-U-E-O vowel targets from Indonesian or English text."""
    if not text or not str(text).strip():
        raise TranscriptError("Transcript is empty")

    lang = _normalize_language(language)
    word_fn = _english_word_syllables if lang == "ENGLISH" else _indonesian_word_syllables

    phrases: list[list[PhonemeTarget]] = []
    for raw_phrase in PHRASE_SPLIT_RE.split(str(text)):
        raw_words = WORD_SPLIT_RE.findall(raw_phrase)
        word_syls: list[list[tuple[str, ...]]] = []
        for raw_word in raw_words:
            # Split hyphenated compounds so each sub-word keeps its own stress pattern.
            for piece in raw_word.split("-"):
                syls = word_fn(piece)
                if syls:
                    word_syls.append(syls)
        if word_syls:
            targets = _build_phrase_targets(word_syls, len(phrases))
            if targets:
                phrases.append(targets)

    if not phrases:
        raise TranscriptError("Transcript contains no extractable vowels")
    return phrases


def _smoothing_kernel(hop_ms: float) -> np.ndarray:
    """Build a symmetric ~40 ms smoothing kernel aligned to hop_ms."""
    width = max(1, int(round(ENVELOPE_WINDOW_MS / hop_ms)))
    if width <= 1:
        return np.array([1.0], dtype=np.float64)
    if width == 4:
        return np.array([0.125, 0.25, 0.25, 0.25, 0.125], dtype=np.float64)
    if width % 2 == 0:
        width += 1
    win = np.hanning(width + 2)[1:-1]
    if np.sum(win) <= 0.0:
        win = np.ones(width, dtype=np.float64)
    return win / np.sum(win)


def _compute_vocal_envelope_db(
    samples: np.ndarray,
    sr: int,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """Compute smoothed 300-3000 Hz vocal energy envelope in dBFS at hop_ms spacing."""
    filtered = audio_dsp.bandpass_vocal(samples, sr, VOCAL_LOW_HZ, VOCAL_HIGH_HZ)
    if filtered.size == 0:
        return np.empty(0, dtype=np.float64)

    hop_len = max(1, int(round(float(sr) * float(hop_ms) / 1000.0)))
    n_samples = filtered.size
    if n_samples <= hop_len:
        n_frames = 1
    else:
        n_frames = max(1, int(round(float(n_samples) / float(hop_len))))

    sq = filtered.astype(np.float64) ** 2
    frame_power = np.empty(n_frames, dtype=np.float64)
    for idx in range(n_frames):
        s = idx * hop_len
        e = min(n_samples, (idx + 1) * hop_len) if idx < n_frames - 1 else n_samples
        if s >= n_samples:
            frame_power[idx] = frame_power[idx - 1]
        else:
            frame_power[idx] = float(np.mean(sq[s:e]))

    frame_db = np.maximum(10.0 * np.log10(np.maximum(frame_power, 1e-20)), -90.0)
    kernel = _smoothing_kernel(float(hop_ms))
    if kernel.size > 1 and frame_db.size > 1:
        pad = kernel.size // 2
        padded = np.pad(frame_db, pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid")[:n_frames]
    return frame_db


def _islands_from_envelope(
    env_db: np.ndarray,
    silence_threshold_db: float,
    hop_ms: float = 10.0,
) -> list[tuple[int, int]]:
    """Segment a smoothed dB envelope into active islands, bridging <= 120 ms gaps."""
    if env_db.size == 0:
        return []

    active = env_db >= float(silence_threshold_db)
    raw_runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            raw_runs.append((start, idx))
            start = None
    if start is not None:
        raw_runs.append((start, int(active.size)))

    if not raw_runs:
        return []

    merged: list[tuple[int, int]] = [raw_runs[0]]
    for s, e in raw_runs[1:]:
        prev_s, prev_e = merged[-1]
        gap_ms = float(s - prev_e) * float(hop_ms)
        if gap_ms <= MAX_GAP_MERGE_MS:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    long_enough = [
        (s, e) for (s, e) in merged if float(e - s) * float(hop_ms) >= MIN_ISLAND_MS
    ]
    return long_enough if long_enough else merged


def detect_speech_islands(
    samples: np.ndarray,
    sr: int,
    silence_threshold_db: float,
    hop_ms: float = 10.0,
) -> list[tuple[int, int]]:
    """Detect active speech intervals [start_hop, end_hop) bridging gaps <= 120 ms."""
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive (got {sr})")
    if hop_ms <= 0:
        raise ValueError(f"hop_ms must be positive (got {hop_ms})")

    env_db = _compute_vocal_envelope_db(samples, sr, hop_ms=hop_ms)
    return _islands_from_envelope(env_db, silence_threshold_db, hop_ms=hop_ms)


def _partition_by_proportions(
    item_weights: list[float],
    target_props: list[float],
    boundary_bonuses: list[float] | None = None,
) -> list[tuple[int, int]]:
    """Partition N items into M contiguous non-empty slices matching target_props."""
    n = len(item_weights)
    m = len(target_props)
    if m <= 1:
        return [(0, n)]
    if m >= n:
        return [(i, i + 1) for i in range(n)]

    total_w = sum(max(w, 1e-6) for w in item_weights)
    pref = [0.0] * (n + 1)
    for idx, w in enumerate(item_weights):
        pref[idx + 1] = pref[idx] + max(w, 1e-6) / total_w

    total_p = sum(max(p, 1e-6) for p in target_props)
    norm_props = [max(p, 1e-6) / total_p for p in target_props]

    inf = 1e18
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    back = [[0] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    for g in range(1, m + 1):
        prop = norm_props[g - 1]
        i_lo = g
        i_hi = n - (m - g)
        for i in range(i_lo, i_hi + 1):
            j_lo = g - 1
            j_hi = i - 1
            best_val = inf
            best_j = j_lo
            for j in range(j_lo, j_hi + 1):
                if dp[g - 1][j] >= inf * 0.5:
                    continue
                frac = pref[i] - pref[j]
                bonus = boundary_bonuses[j] if (boundary_bonuses is not None and j > 0) else 0.0
                cand = dp[g - 1][j] + (frac - prop) ** 2 - bonus
                if cand < best_val:
                    best_val = cand
                    best_j = j
            dp[g][i] = best_val
            back[g][i] = best_j

    slices: list[tuple[int, int]] = []
    cur_end = n
    for g in range(m, 0, -1):
        prev_end = back[g][cur_end]
        slices.append((prev_end, cur_end))
        cur_end = prev_end
    slices.reverse()
    return slices


def _island_energy_mass(island: tuple[int, int], env_db: np.ndarray) -> float:
    """Compute energy-weighted duration proxy for matching phrases to islands."""
    s, e = island
    if e <= s:
        return 1e-6
    lin = np.power(10.0, env_db[s:e] / 20.0)
    return max(1e-6, float(np.sum(lin)))


def _match_phrases_to_islands(
    phrases: list[list[PhonemeTarget]],
    islands: list[tuple[int, int]],
    env_db: np.ndarray,
) -> list[tuple[int, tuple[int, int], list[PhonemeTarget]]]:
    """Match punctuation phrases to detected speech islands, returning (island_idx, span, targets)."""
    p_count = len(phrases)
    m_count = len(islands)

    if p_count == m_count:
        return [(idx, islands[idx], phrases[idx]) for idx in range(m_count)]

    phrase_weights = [sum(t.weight for t in ph) for ph in phrases]
    island_masses = [_island_energy_mass(isl, env_db) for isl in islands]

    if m_count < p_count:
        # Multiple punctuation phrases fell inside fewer speech islands: merge phrases into M groups.
        phrase_slices = _partition_by_proportions(phrase_weights, island_masses)
        assignments: list[tuple[int, tuple[int, int], list[PhonemeTarget]]] = []
        for isl_idx, (ps, pe) in enumerate(phrase_slices):
            merged_targets = [t for ph in phrases[ps:pe] for t in ph]
            assignments.append((isl_idx, islands[isl_idx], merged_targets))
        return assignments

    # More speech islands than phrases: group islands to phrases, then split targets cleanly.
    island_slices = _partition_by_proportions(island_masses, phrase_weights)
    assignments = []
    for p_idx, (is_s, is_e) in enumerate(island_slices):
        sub_islands = islands[is_s:is_e]
        targets = phrases[p_idx]
        if len(sub_islands) == 1:
            assignments.append((is_s, sub_islands[0], targets))
            continue

        if len(targets) >= len(sub_islands):
            t_weights = [t.weight for t in targets]
            bonuses = [0.0] * (len(targets) + 1)
            for idx in range(1, len(targets)):
                if not targets[idx].same_as_prev_in_word:
                    bonuses[idx] = 0.01
            sub_masses = [_island_energy_mass(isl, env_db) for isl in sub_islands]
            t_slices = _partition_by_proportions(t_weights, sub_masses, boundary_bonuses=bonuses)
            for offset, (ts, te) in enumerate(t_slices):
                assignments.append((is_s + offset, sub_islands[offset], targets[ts:te]))
        else:
            # Fewer targets than sub-islands: bind targets to strongest sub-island.
            best_offset = max(
                range(len(sub_islands)),
                key=lambda k: _island_energy_mass(sub_islands[k], env_db),
            )
            assignments.append((is_s + best_offset, sub_islands[best_offset], targets))
    return assignments


def _frame_index_for_hop(
    hop_idx: int,
    hop_ms: float,
    frames: list[audio_dsp.FormantFrame],
) -> int:
    """Map 10ms hop frame index to nearest FormantFrame index."""
    if not frames:
        return 0
    if len(frames) == 1:
        return 0
    center_sec = (float(hop_idx) + 0.5) * (hop_ms / 1000.0)
    t0 = frames[0].time_sec
    dt = (frames[-1].time_sec - t0) / float(len(frames) - 1)
    if dt <= 0.0:
        return 0
    return min(len(frames) - 1, max(0, int(round((center_sec - t0) / dt))))


def _span_peak_rms_db(
    s_hop: int,
    e_hop: int,
    hop_ms: float,
    frames: list[audio_dsp.FormantFrame],
    env_db: np.ndarray,
) -> float:
    """Compute peak RMS dB inside [s_hop, e_hop) using raw FormantFrames with envelope fallback."""
    if frames:
        f_lo = _frame_index_for_hop(s_hop, hop_ms, frames)
        f_hi = _frame_index_for_hop(max(s_hop, e_hop - 1), hop_ms, frames)
        if f_hi >= f_lo:
            return max(frames[i].rms_db for i in range(f_lo, f_hi + 1))
    if e_hop > s_hop:
        return float(np.max(env_db[s_hop:e_hop]))
    return float(env_db[min(s_hop, env_db.size - 1)])


def _align_island_targets(
    island_idx: int,
    island: tuple[int, int],
    targets: list[PhonemeTarget],
    env_db: np.ndarray,
    frames: list[audio_dsp.FormantFrame],
    norm_scores: list[dict[str, float]],
    hop_ms: float,
    fps: float,
    start_frame: int,
) -> list[tuple[int, tuple[int, int], PhonemeTarget, AlignedSyllable]]:
    """Run constrained monotonic DTW inside one speech island for its targets."""
    s_idx, e_idx = island
    n_frames = e_idx - s_idx
    k_count = len(targets)
    hop_sec = hop_ms / 1000.0

    if k_count == 1:
        peak_db = _span_peak_rms_db(s_idx, e_idx, hop_ms, frames, env_db)
        syl = AlignedSyllable(
            vowel=targets[0].vowel,
            start_frame=float(start_frame) + float(s_idx) * hop_sec * float(fps),
            end_frame=float(start_frame) + float(e_idx) * hop_sec * float(fps),
            peak_rms_db=peak_db,
            valley_dip_before=False,
        )
        return [(island_idx, (s_idx, e_idx), targets[0], syl)]

    if n_frames < k_count:
        # Very short island (< K frames): distribute duration proportionally by weights.
        total_w = sum(max(t.weight, 1e-4) for t in targets)
        t0 = float(s_idx) * hop_sec
        dur = max(1e-4, float(n_frames) * hop_sec)
        cur_t = t0
        out: list[tuple[int, tuple[int, int], PhonemeTarget, AlignedSyllable]] = []
        peak_db = _span_peak_rms_db(s_idx, e_idx, hop_ms, frames, env_db)
        for idx, tgt in enumerate(targets):
            frac = max(tgt.weight, 1e-4) / total_w
            nxt_t = t0 + dur if idx == k_count - 1 else cur_t + dur * frac
            hop_lo = min(e_idx - 1, s_idx + int(math.floor((cur_t - t0) / hop_sec)))
            hop_hi = max(hop_lo + 1, min(e_idx, s_idx + int(math.ceil((nxt_t - t0) / hop_sec))))
            syl = AlignedSyllable(
                vowel=tgt.vowel,
                start_frame=float(start_frame) + cur_t * float(fps),
                end_frame=float(start_frame) + nxt_t * float(fps),
                peak_rms_db=peak_db,
                valley_dip_before=False,
            )
            out.append((island_idx, (hop_lo, hop_hi), tgt, syl))
            cur_t = nxt_t
        return out

    # Extract per-frame acoustic energy and valley depth within the island.
    env_lin = np.power(10.0, env_db[s_idx:e_idx] / 20.0)
    e_max = max(float(np.max(env_lin)), 1e-9)
    e_norm = env_lin / e_max

    look = 25
    valley_depth = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        l_max = float(np.max(e_norm[max(0, t - look) : t + 1]))
        r_max = float(np.max(e_norm[t : min(n_frames, t + look + 1)]))
        valley_depth[t] = max(0.0, min(l_max, r_max) - float(e_norm[t]))

    switch_cost = 2.5 * e_norm + 1.0 * (1.0 - valley_depth)

    # Expected relative center and generous Sakoe-Chiba band per target.
    weights = [max(t.weight, 1e-4) for t in targets]
    total_w = sum(weights)
    cum = 0.0
    mu = np.empty(k_count, dtype=np.float64)
    c_bounds = np.empty((k_count, 2), dtype=np.float64)
    for k, w in enumerate(weights):
        c0 = cum / total_w
        cum += w
        c1 = cum / total_w
        c_bounds[k, 0] = c0
        c_bounds[k, 1] = c1
        mu[k] = 0.5 * (c0 + c1)

    slack = max(0.45, 2.5 / float(k_count))
    t_lo = np.empty(k_count, dtype=np.int32)
    t_hi = np.empty(k_count, dtype=np.int32)
    for k in range(k_count):
        lo = max(k, int(math.floor((c_bounds[k, 0] - slack) * n_frames)))
        hi = min(n_frames - (k_count - k), int(math.ceil((c_bounds[k, 1] + slack) * n_frames)) - 1)
        lo = min(lo, n_frames - (k_count - k))
        hi = max(hi, lo)
        t_lo[k] = lo
        t_hi[k] = hi

    # Precompute frame emission costs (speaker-normalized formants + position prior).
    frame_cost = np.empty((k_count, n_frames), dtype=np.float64)
    u_grid = np.linspace(0.0, 1.0, n_frames) if n_frames > 1 else np.zeros(1, dtype=np.float64)
    for t in range(n_frames):
        f_idx = _frame_index_for_hop(s_idx + t, hop_ms, frames)
        if frames and frames[f_idx].is_voiced and norm_scores:
            sc_map = norm_scores[f_idx]
            sc_sum = sum(sc_map.values())
        else:
            sc_map = None
            sc_sum = 0.0

        u_val = float(u_grid[t])
        e_val = float(e_norm[t])
        for k in range(k_count):
            pos_term = 1.5 * ((u_val - float(mu[k])) ** 2)
            if sc_map is not None and sc_sum > 0.0:
                prob = max(float(sc_map.get(targets[k].vowel, 0.01)), 0.01)
                formant_term = 1.2 * e_val * (-math.log(prob))
            else:
                formant_term = 0.0
            frame_cost[k, t] = formant_term + pos_term

    inf = 1e18
    dp = np.full((k_count, n_frames), inf, dtype=np.float64)
    back = np.zeros((k_count, n_frames), dtype=np.int8)
    dp[0, 0] = float(frame_cost[0, 0])

    for t in range(1, n_frames):
        sw = float(switch_cost[t])
        for k in range(k_count):
            if t < t_lo[k] or t > t_hi[k]:
                continue
            best = float(dp[k, t - 1])
            step = 0
            if k > 0:
                cand = float(dp[k - 1, t - 1]) + sw
                if cand < best:
                    best = cand
                    step = 1
            if best < inf * 0.5:
                dp[k, t] = best + float(frame_cost[k, t])
                back[k, t] = step

    # Backtrack frame assignments from (k_count - 1, n_frames - 1) to (0, 0).
    assigned_k = [0] * n_frames
    cur_k = k_count - 1
    for t in range(n_frames - 1, -1, -1):
        assigned_k[t] = cur_k
        if t > 0 and back[cur_k, t] == 1:
            cur_k -= 1

    spans: list[tuple[int, int]] = []
    run_start = 0
    for t in range(1, n_frames):
        if assigned_k[t] != assigned_k[t - 1]:
            spans.append((s_idx + run_start, s_idx + t))
            run_start = t
    spans.append((s_idx + run_start, e_idx))

    out = []
    for k, (hop_lo, hop_hi) in enumerate(spans):
        peak_db = _span_peak_rms_db(hop_lo, hop_hi, hop_ms, frames, env_db)
        syl = AlignedSyllable(
            vowel=targets[k].vowel,
            start_frame=float(start_frame) + float(hop_lo) * hop_sec * float(fps),
            end_frame=float(start_frame) + float(hop_hi) * hop_sec * float(fps),
            peak_rms_db=peak_db,
            valley_dip_before=False,
        )
        out.append((island_idx, (hop_lo, hop_hi), targets[k], syl))
    return out


def align_transcript(
    samples: np.ndarray,
    sr: int,
    transcript: str,
    language: str = "INDONESIAN",
    silence_threshold_db: float = -38.0,
    fps: float = 24.0,
    start_frame: int = 1,
) -> list[AlignedSyllable]:
    """Align extracted transcript vowels to acoustic speech islands using DTW."""
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive (got {sr})")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive (got {fps})")

    lang = _normalize_language(language)
    phrases = extract_vowels(transcript, lang)

    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return []

    work_sr = audio_dsp.TARGET_SAMPLE_RATE
    if int(sr) != work_sr:
        mono_16k = audio_dsp.resample_mono(arr, int(sr), work_sr)
    else:
        mono_16k = arr

    hop_ms = audio_dsp.HOP_MS
    env_db = _compute_vocal_envelope_db(mono_16k, work_sr, hop_ms=hop_ms)
    islands = _islands_from_envelope(env_db, silence_threshold_db, hop_ms=hop_ms)
    if not islands:
        return []

    frames = audio_dsp.analyze_audio_frames(
        mono_16k,
        sr=work_sr,
        silence_threshold_db=silence_threshold_db,
        voice_type="FEMALE_STANDARD",
        language=lang,
    )
    norm_scores = audio_dsp.compute_speaker_normalized_scores(frames)

    matches = _match_phrases_to_islands(phrases, islands, env_db)

    entries: list[tuple[int, tuple[int, int], PhonemeTarget, AlignedSyllable]] = []
    for island_idx, island, targets in matches:
        entries.extend(
            _align_island_targets(
                island_idx=island_idx,
                island=island,
                targets=targets,
                env_db=env_db,
                frames=frames,
                norm_scores=norm_scores,
                hop_ms=hop_ms,
                fps=fps,
                start_frame=start_frame,
            )
        )

    for idx in range(1, len(entries)):
        prev_island, _, _, prev_syl = entries[idx - 1]
        cur_island, _, cur_target, cur_syl = entries[idx]
        if cur_syl.vowel == prev_syl.vowel and (
            cur_island == prev_island or cur_target.same_as_prev_in_word
        ):
            cur_syl.valley_dip_before = True

    return [entry[3] for entry in entries]
