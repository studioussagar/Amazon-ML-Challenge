from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import HashingVectorizer

from . import config
from .block_keys import match_text, name_prefix
from .text_normalize import normalize_address, normalize_country, normalize_name

# Standard Soundex translation table
_DIGITS_MAP = {
    "B": "1", "F": "1", "P": "1", "V": "1",
    "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
    "D": "3", "T": "3", "L": "4", "M": "5", "N": "5", "R": "6",
}


def fast_soundex(name: str) -> str:
    """Compute 4-character phonetic soundex code."""
    if not name:
        return "0000"
    clean = "".join(c for c in name.upper() if c.isalpha())
    if not clean:
        return "0000"
    first = clean[0]
    tail = [_DIGITS_MAP.get(c, "0") for c in clean[1:]]
    res = [first]
    prev = _DIGITS_MAP.get(first, "0")
    for d in tail:
        if d != "0" and d != prev:
            res.append(d)
        prev = d
    return "".join(res)[:4].ljust(4, "0")


def token_jaccard(toks_a: Sequence[str], toks_b: Sequence[str]) -> float:
    """Jaccard index between two sets of tokens."""
    set_a = set(toks_a)
    set_b = set(toks_b)
    if not set_a and not set_b:
        return 1.0
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 0.0
    return len(set_a & set_b) / union_len


def compute_features_batch(
    s1_names: List[str],
    s1_addrs: List[str],
    s1_countries: List[str],
    cand_names: List[str],
    cand_addrs: List[str],
    cand_countries: List[str],
) -> Dict[str, np.ndarray]:
    """Compute pairwise similarity metrics across candidate batches."""
    n = len(s1_names)
    f_lev = np.zeros(n, dtype=np.float32)
    f_jw = np.zeros(n, dtype=np.float32)
    f_name_jaccard = np.zeros(n, dtype=np.float32)
    f_addr_jaccard = np.zeros(n, dtype=np.float32)
    f_phonetic = np.zeros(n, dtype=np.int8)
    f_country = np.zeros(n, dtype=np.int8)
    f_len_diff = np.zeros(n, dtype=np.float32)

    # Compute string similarities and token overlaps
    for i in range(n):
        na, nb = s1_names[i], cand_names[i]
        f_lev[i] = Levenshtein.normalized_similarity(na, nb)
        f_jw[i] = JaroWinkler.similarity(na, nb)

        tok_na = na.split()
        tok_nb = nb.split()
        f_name_jaccard[i] = token_jaccard(tok_na, tok_nb)

        tok_aa = s1_addrs[i].split()
        tok_ab = cand_addrs[i].split()
        f_addr_jaccard[i] = token_jaccard(tok_aa, tok_ab)

        f_phonetic[i] = 1 if fast_soundex(na) == fast_soundex(nb) else 0
        f_country[i] = 1 if s1_countries[i] == cand_countries[i] else 0

        max_l = max(len(na), len(nb), 1)
        f_len_diff[i] = abs(len(na) - len(nb)) / max_l

    # Character n-gram hashing vectorizer for fast cosine similarity without vocab overhead
    vec = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        n_features=2048,
        alternate_sign=False,
        norm="l2",
        dtype=np.float32,
    )
    s1_texts = [f"{n} {a}" for n, a in zip(s1_names, s1_addrs)]
    cand_texts = [f"{n} {a}" for n, a in zip(cand_names, cand_addrs)]
    m1 = vec.transform(s1_texts)
    m2 = vec.transform(cand_texts)
    f_tfidf = np.asarray(m1.multiply(m2).sum(axis=1), dtype=np.float32).ravel()

    return {
        "name_levenshtein": f_lev,
        "name_jaro_winkler": f_jw,
        "name_token_jaccard": f_name_jaccard,
        "addr_token_jaccard": f_addr_jaccard,
        "tfidf_cosine": f_tfidf,
        "phonetic_match": f_phonetic,
        "country_match": f_country,
        "length_diff": f_len_diff,
    }
