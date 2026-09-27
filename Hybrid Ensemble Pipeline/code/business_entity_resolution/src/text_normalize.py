from __future__ import annotations

import re
import unicodedata
from typing import Dict

# Standardize common legal business suffixes across target regions
LEGAL_SUFFIX_MAP: Dict[str, str] = {
    "pvt": "private",
    "pvt.": "private",
    "pvts": "private",
    "pvtltd": "private",
    "p": "private",
    "ltd": "limited",
    "ltds": "limited",
    "ltda": "limited",
    "corp": "corporation",
    "corps": "corporation",
    "crop": "corporation",  # common typo
    "inc": "incorporated",
    "incs": "incorporated",
    "co": "company",
    "company": "company",
    "llp": "limited liability partnership",
    "llc": "limited liability company",
    "lllp": "limited liability limited partnership",
    "lp": "limited partnership",
    "plc": "public limited company",
    "pt": "private limited",
    "sarl": "societe a responsabilite limitee",
    "sa": "societe anonyme",
    "sas": "societe par actions simplifiee",
    "eurl": "entreprise unipersonnelle a responsabilite limitee",
    "dba": "doing business as",
    "t/a": "trading as",
}

_NAME_NOISE_TOKENS: frozenset[str] = frozenset({"--"})

# Standard street types, unit markers, and landmarks
ADDRESS_ABBREV_MAP: Dict[str, str] = {
    "rd": "road",
    "rd.": "road",
    "st": "street",
    "st.": "street",
    "ave": "avenue",
    "ave.": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "dr.": "drive",
    "ct": "court",
    "ct.": "court",
    "pl": "place",
    "pl.": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "cir": "circle",
    "ter": "terrace",
    "trl": "trail",
    "tpke": "turnpike",
    "expy": "expressway",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "apt": "apartment",
    "apt.": "apartment",
    "ste": "suite",
    "bldg": "building",
    "fl": "floor",
    "no": "number",
    "no.": "number",
    "kh": "khasra",
    "pin": "pincode",
    "po": "post office",
    "sector": "sector",
    "near": "near",
    "nre": "near",
    "opp": "opposite",
    "opp.": "opposite",
    "behind": "behind",
    "b/h": "behind",
    "bldng": "building",
}

_ADDRESS_KEEP_TOKENS: frozenset[str] = frozenset({"near", "opposite", "behind"})

# Keep letters, numbers, and combining marks (preserves Indic matras)
_LEXEME_CLASSES: frozenset[str] = frozenset(("L", "N", "M"))

_TOKENIZE_TABLE = str.maketrans(
    {
        cp: " "
        for cp in range(0x10000)
        if not (
            unicodedata.category(chr(cp))[0] in _LEXEME_CLASSES
            or chr(cp).isspace()
        )
    }
)

_ASCII_WORD_RE = re.compile(r"[0-9A-Za-z]+")
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_ORDINAL_RE = re.compile(r"(\d+)(st|nd|rd|th)\b", flags=re.IGNORECASE)
_NUM_LETTER_SPLIT_RE = re.compile(r"(?<=\d)(?=[a-zA-Z])", flags=re.UNICODE)
_LETTER_NUM_SPLIT_RE = re.compile(r"(?<=[a-zA-Z])(?=\d)", flags=re.UNICODE)


def fold_accents(text: str) -> str:
    """Fold Latin accents to ASCII while keeping Indic scripts intact."""
    out_chars = []
    for ch in text:
        decomposed = unicodedata.normalize("NFKD", ch)
        if unicodedata.combining(ch):
            out_chars.append(ch)
        else:
            out_chars.append(decomposed)
    rebuilt = "".join(out_chars)

    result = []
    prev_is_ascii_alpha = False
    for ch in unicodedata.normalize("NFKD", rebuilt):
        if unicodedata.combining(ch):
            # Only drop combining diacritics if attached to Latin ASCII characters
            if prev_is_ascii_alpha:
                continue
            result.append(ch)
        else:
            result.append(ch)
            prev_is_ascii_alpha = ch.isascii() and ch.isalpha()
    return unicodedata.normalize("NFC", "".join(result))


def normalize(text: object) -> str:
    """Clean raw text string: NFKC, lowercased, punctuation removed."""
    if text is None:
        return ""
    if isinstance(text, float) and text != text:
        return ""

    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    s = fold_accents(s)
    s = s.casefold()

    # Convert ordinals like 41st -> 41th so 'st' isn't mistaken for 'street'
    s = _ORDINAL_RE.sub(r"\1th", s)
    # Separate glued digits and letters (e.g., 5bis -> 5 bis)
    s = _NUM_LETTER_SPLIT_RE.sub(" ", s)
    s = _LETTER_NUM_SPLIT_RE.sub(" ", s)

    if s.isascii():
        return " ".join(_ASCII_WORD_RE.findall(s))
    return " ".join(s.translate(_TOKENIZE_TABLE).split())


def _apply_token_map(tokens: list[str], mapping: Dict[str, str]) -> list[str]:
    expanded: list[str] = []
    for tok in tokens:
        replacement = mapping.get(tok)
        if replacement is None:
            expanded.append(tok)
        else:
            expanded.extend(replacement.split())
    return expanded


def normalize_name(text: object) -> str:
    """Normalize business name with legal suffix expansions."""
    base = normalize(text)
    if not base:
        return ""
    tokens = base.split()
    tokens = _apply_token_map(tokens, LEGAL_SUFFIX_MAP)
    filtered = [t for t in tokens if t not in _NAME_NOISE_TOKENS]
    return " ".join(filtered) if filtered else " ".join(tokens)


def normalize_address(text: object) -> str:
    """Normalize address with street and landmark term expansions."""
    base = normalize(text)
    if not base:
        return ""
    tokens = base.split()
    tokens = _apply_token_map(tokens, ADDRESS_ABBREV_MAP)
    return " ".join(tokens)


def normalize_country(text: object) -> str:
    """Normalize country code/name."""
    if text is None:
        return ""
    s = str(text).strip()
    if not s or (isinstance(text, float) and text != text):
        return ""
    return unicodedata.normalize("NFKC", s).casefold()


def name_tokens(text: object) -> list[str]:
    normalized = normalize_name(text)
    return normalized.split() if normalized else []


def address_tokens(text: object) -> list[str]:
    normalized = normalize_address(text)
    return normalized.split() if normalized else []
