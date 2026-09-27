from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    from . import config
    from .text_normalize import normalize_address, normalize_country, normalize_name
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config
    from src.text_normalize import (
        normalize_address,
        normalize_country,
        normalize_name,
    )

# Bounded in-memory cache to avoid redundant text normalization across millions of rows
_NORM_CACHE_MAX = 500_000
_name_cache: dict[str, str] = {}
_addr_cache: dict[str, str] = {}
_country_cache: dict[str, str] = {}


def _cache_lookup(cache: dict, value: object, fn) -> str:
    key = "" if value is None else str(value)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = fn(key)
    if len(cache) >= _NORM_CACHE_MAX:
        cache.clear()
    cache[key] = result
    return result


def clear_norm_caches() -> None:
    """Clear normalization caches to reclaim heap space."""
    _name_cache.clear()
    _addr_cache.clear()
    _country_cache.clear()


def cached_normalize_name(value: object) -> str:
    return _cache_lookup(_name_cache, value, normalize_name)


def cached_normalize_address(value: object) -> str:
    return _cache_lookup(_addr_cache, value, normalize_address)


def cached_normalize_country(value: object) -> str:
    return _cache_lookup(_country_cache, value, normalize_country)


def load_source(path: Path | str, normalize: bool = True) -> pd.DataFrame:
    """Load raw entity TSV file and apply normalization."""
    path = Path(path)
    # Using C engine with QUOTE_NONE to avoid mangling unquoted quotes in names
    df = pd.read_csv(
        path,
        sep=config.SEP,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        encoding="utf-8",
        engine="c",
        quoting=csv.QUOTE_NONE,
    )

    missing = [c for c in config.RECORD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required column(s) {missing}")
    df = df[config.RECORD_COLUMNS].copy()

    for col in config.RECORD_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    if normalize:
        df["source"] = df["entity_id"].str[:3].str.rstrip("-")
        df["name_norm"] = df["business_name"].map(cached_normalize_name)
        df["addr_norm"] = df["business_address"].map(cached_normalize_address)
        df["country_key"] = df["country"].map(cached_normalize_country)

    return df


def load_ground_truth(path: Path | str) -> pd.DataFrame:
    """Load ground truth matches TSV."""
    path = Path(path)
    df = pd.read_csv(
        path,
        sep=config.SEP,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        encoding="utf-8",
        engine="c",
        quoting=csv.QUOTE_NONE,
    )
    missing = [c for c in config.GROUND_TRUTH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required column(s) {missing}")

    df = df[config.GROUND_TRUTH_COLUMNS].copy()
    df["source1_entity_id"] = df["source1_entity_id"].str.strip()
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").astype(str).str.strip()
    df["matched_list"] = df["matched_entity_ids"].map(
        lambda s: [x for x in s.split(",") if x] if s else []
    )
    return df


def quick_profile(df: pd.DataFrame, label: str = "df") -> Dict[str, object]:
    """Inspect dataset statistics and country distributions."""
    n = len(df)
    n_name_empty = int((df["business_name"] == "").sum()) if "business_name" in df else 0
    n_addr_empty = int((df["business_address"] == "").sum()) if "business_address" in df else 0
    n_country_empty = int((df["country"] == "").sum()) if "country" in df else 0

    countries = df["country"].value_counts().to_dict() if "country" in df else {}
    dup_ids = int(df["entity_id"].duplicated().sum()) if "entity_id" in df else 0

    profile: Dict[str, object] = {
        "label": label,
        "rows": n,
        "unique_entity_ids": int(df["entity_id"].nunique()) if "entity_id" in df else 0,
        "duplicate_entity_ids": dup_ids,
        "empty_business_name": n_name_empty,
        "empty_business_address": n_addr_empty,
        "empty_country": n_country_empty,
        "countries": countries,
    }

    print(f"\n=== profile: {label} ===")
    print(f"  rows                  : {profile['rows']:,}")
    print(f"  unique entity_ids     : {profile['unique_entity_ids']:,}")
    print(f"  duplicate entity_ids  : {profile['duplicate_entity_ids']:,}")
    print(f"  empty business_name   : {profile['empty_business_name']:,}")
    print(f"  empty business_address: {profile['empty_business_address']:,}")
    print(f"  empty country         : {profile['empty_country']:,}")
    print("  country distribution  :")
    for c, cnt in sorted(countries.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * cnt / n if n else 0.0
        shown = c if c else "<empty>"
        print(f"      {shown:<12} {cnt:>10,}  ({pct:5.1f}%)")
    return profile


def profile_ground_truth(gt: pd.DataFrame) -> Dict[str, object]:
    """Inspect ground truth cardinality and multi-match counts."""
    counts = gt["matched_list"].map(len)
    singleton = int((counts == 0).sum())
    has_match = int((counts > 0).sum())

    n_s2 = sum(1 for lst in gt["matched_list"] for x in lst if x.startswith("S2-"))
    n_s3 = sum(1 for lst in gt["matched_list"] for x in lst if x.startswith("S3-"))
    both = int(
        gt["matched_list"].map(
            lambda lst: any(x.startswith("S2-") for x in lst)
            and any(x.startswith("S3-") for x in lst)
        ).sum()
    )

    profile: Dict[str, object] = {
        "rows": len(gt),
        "singletons": singleton,
        "with_match": has_match,
        "total_matched_ids": n_s2 + n_s3,
        "matched_from_s2": n_s2,
        "matched_from_s3": n_s3,
        "entities_matching_both_sources": both,
        "max_matches_for_one_entity": int(counts.max()) if len(counts) else 0,
    }

    print("\n=== profile: train_ground_truth ===")
    print(f"  rows                       : {profile['rows']:,}")
    print(f"  singletons (no match)      : {profile['singletons']:,}")
    print(f"  entities with >=1 match    : {profile['with_match']:,}")
    print(f"  total matched IDs          : {profile['total_matched_ids']:,}")
    print(f"    - from Source 2          : {profile['matched_from_s2']:,}")
    print(f"    - from Source 3          : {profile['matched_from_s3']:,}")
    print(f"  entities matching BOTH srcs: {profile['entities_matching_both_sources']:,}")
    print(f"  max matches for one entity : {profile['max_matches_for_one_entity']:,}")
    return profile


def load_train(profile: bool = True, nrows: Optional[int] = None) -> Dict[str, object]:
    """Load training splits and ground truth."""
    if nrows is not None:
        frames = {}
        for key, path in (
            ("source1", config.TRAIN_SOURCE1),
            ("source2", config.TRAIN_SOURCE2),
            ("source3", config.TRAIN_SOURCE3),
        ):
            raw = pd.read_csv(
                path, sep=config.SEP, dtype=str, keep_default_na=False,
                na_values=[], encoding="utf-8", engine="c",
                quoting=csv.QUOTE_NONE, nrows=nrows,
            )
            frames[key] = _postprocess(raw)
        gt = load_ground_truth(config.TRAIN_GROUND_TRUTH).head(nrows)
    else:
        frames = {
            "source1": load_source(config.TRAIN_SOURCE1),
            "source2": load_source(config.TRAIN_SOURCE2),
            "source3": load_source(config.TRAIN_SOURCE3),
        }
        gt = load_ground_truth(config.TRAIN_GROUND_TRUTH)

    result: Dict[str, object] = {**frames, "ground_truth": gt}

    if profile:
        for key in ("source1", "source2", "source3"):
            quick_profile(frames[key], f"train_{key}")
        profile_ground_truth(gt)

    return result


def load_test(profile: bool = True, nrows: Optional[int] = None) -> Dict[str, object]:
    """Load test splits."""
    if nrows is not None:
        frames = {}
        for key, path in (
            ("source1", config.TEST_SOURCE1),
            ("source2", config.TEST_SOURCE2),
            ("source3", config.TEST_SOURCE3),
        ):
            raw = pd.read_csv(
                path, sep=config.SEP, dtype=str, keep_default_na=False,
                na_values=[], encoding="utf-8", engine="c",
                quoting=csv.QUOTE_NONE, nrows=nrows,
            )
            frames[key] = _postprocess(raw)
    else:
        frames = {
            "source1": load_source(config.TEST_SOURCE1),
            "source2": load_source(config.TEST_SOURCE2),
            "source3": load_source(config.TEST_SOURCE3),
        }

    result: Dict[str, object] = dict(frames)

    if profile:
        for key in ("source1", "source2", "source3"):
            quick_profile(frames[key], f"test_{key}")
        if nrows is not None:
            train_countries = set(
                pd.read_csv(
                    config.TRAIN_SOURCE1, sep=config.SEP, usecols=["country"],
                    dtype=str, keep_default_na=False, na_values=[],
                    encoding="utf-8", engine="c", quoting=csv.QUOTE_NONE,
                    nrows=nrows,
                )["country"].str.strip().unique()
            )
        else:
            train_countries = set(
                load_source(config.TRAIN_SOURCE1)[["country"]].squeeze("columns").unique()
            )
        test_countries = set(frames["source1"]["country"].unique())
        print("\n=== country coverage ===")
        print(f"  train countries : {sorted(train_countries)}")
        print(f"  test countries  : {sorted(test_countries)}")

    return result


def _postprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[config.RECORD_COLUMNS].copy()
    for col in config.RECORD_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    df["source"] = df["entity_id"].str[:3].str.rstrip("-")
    df["name_norm"] = df["business_name"].map(cached_normalize_name)
    df["addr_norm"] = df["business_address"].map(cached_normalize_address)
    df["country_key"] = df["country"].map(cached_normalize_country)
    return df


if __name__ == "__main__":
    import argparse

    config.enable_utf8_stdout()

    parser = argparse.ArgumentParser(description="Data loader")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Load only first N rows")
    args = parser.parse_args()

    if args.split in ("train", "both"):
        load_train(profile=True, nrows=args.nrows)
    if args.split in ("test", "both"):
        load_test(profile=True, nrows=args.nrows)
