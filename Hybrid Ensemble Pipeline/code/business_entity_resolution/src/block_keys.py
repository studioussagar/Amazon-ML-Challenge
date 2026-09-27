from __future__ import annotations

import re

from . import config
from .data_loader import (
    cached_normalize_address,
    cached_normalize_country,
    cached_normalize_name,
)
from .text_normalize import normalize_address, normalize_name

# Using 3-char prefixes balances candidate pool sizes and recall
PREFIX_LEN: int = 3
EMPTY_PREFIX: str = "__empty__"

_WS_RE = re.compile(r"\s+")


def name_prefix(name_norm: str, prefix_len: int = PREFIX_LEN) -> str:
    """Extract first N spaceless characters of business name."""
    compact = _WS_RE.sub("", name_norm or "")
    if not compact:
        return EMPTY_PREFIX
    return compact[:prefix_len]


# Common leading prefixes and noise words across US, India, and France
LEAD_NOISE_RE = re.compile(
    r"^(the|shri|sri|dr|dba|inc|mr|ms|m/s|m\s*s|center|national|all|new|om)\s+",
    re.IGNORECASE,
)
NUM_RE = re.compile(r"\b\d+\b")
ADDR_WORD_RE = re.compile(r"\b[a-zA-Z]{4,}\b")

# High-frequency address stopwords that do not help disambiguation
STOP_ADDR = {
    "road", "street", "near", "nagar", "opp", "opposite", "behind", "floor",
    "building", "plot", "sector", "cross", "main", "avenue", "lane", "circle",
    "layout", "complex", "house", "shop", "village", "dist", "district",
    "state", "bazaar", "market", "west", "east", "north", "south", "city",
    "block", "bldg", "flr", "apt", "suite", "room", "dept", "care", "attn",
    "post", "office", "area", "colony", "marg", "puram", "chowk", "highway",
    "number", "side", "front", "phase", "stage", "line", "ward",
}


def stripped_prefix(name_norm: str, prefix_len: int = PREFIX_LEN) -> str:
    """Prefix after removing leading honorifics or noise words."""
    stripped = LEAD_NOISE_RE.sub("", name_norm or "").strip()
    return name_prefix(stripped, prefix_len)


def addr_key(addr_norm: str) -> str:
    """Selective address key combining house/plot number and street token."""
    nums = NUM_RE.findall(addr_norm or "")
    num = nums[0] if nums else ""
    words = [w for w in ADDR_WORD_RE.findall(addr_norm or "") if w not in STOP_ADDR]
    w = words[0] if words else ""
    if num and w:
        return f"{num}_{w}"
    if num:
        return f"n_{num}"
    if w:
        return f"w_{w}"
    return "__none__"


def word2_key(name_norm: str, prefix_len: int = PREFIX_LEN) -> str:
    """Prefix of the second token to catch inverted word order."""
    words = (name_norm or "").split()
    return words[1][:prefix_len] if len(words) > 1 else "__none__"


def block_key(country: object, name_norm: str, prefix_len: int = PREFIX_LEN) -> tuple[str, str]:
    return (config.normalize_country_key(country), name_prefix(name_norm, prefix_len))


def match_text(name_norm: str, addr_norm: str) -> str:
    """Combined text representation used for similarity matching."""
    if name_norm and addr_norm:
        return f"{name_norm} {addr_norm}"
    return name_norm or addr_norm or ""


def add_block_columns(df, prefix_len: int = PREFIX_LEN):
    """Add normalized fields and blocking key columns to dataframe."""
    try:
        import polars as pl
    except ImportError:
        pl = None

    if pl is not None and isinstance(df, pl.DataFrame):
        rows = df.to_dicts()
        for r in rows:
            nn = cached_normalize_name(r.get("business_name", ""))
            an = cached_normalize_address(r.get("business_address", ""))
            ck = config.normalize_country_key(cached_normalize_country(r.get("country", "")))
            r["name_norm"] = nn
            r["addr_norm"] = an
            r["country_key"] = ck
            r["block_key"] = f"{ck}\x1f{name_prefix(nn, prefix_len)}"
            r["match_text"] = match_text(nn, an)
        return pl.DataFrame(rows)

    df = df.copy()
    df["name_norm"] = df["business_name"].map(cached_normalize_name)
    df["addr_norm"] = df["business_address"].map(cached_normalize_address)
    df["country_key"] = df["country"].map(
        lambda c: config.normalize_country_key(cached_normalize_country(c))
    )
    df["block_key"] = (
        df["country_key"] + "\x1f" + df["name_norm"].map(lambda s: name_prefix(s, prefix_len))
    )
    df["match_text"] = [
        match_text(n, a) for n, a in zip(df["name_norm"], df["addr_norm"])
    ]
    return df
