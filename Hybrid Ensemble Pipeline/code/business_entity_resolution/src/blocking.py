from __future__ import annotations

import gc
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl

try:
    from . import config
    from .block_keys import addr_key, match_text, name_prefix, stripped_prefix, word2_key
    from .blocking_embedding import build_candidates_embedding
    from .blocking_tfidf import build_candidates_tfidf
    from .data_loader import (
        cached_normalize_address,
        cached_normalize_country,
        cached_normalize_name,
        clear_norm_caches,
        load_ground_truth,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config
    from src.block_keys import addr_key, match_text, name_prefix, stripped_prefix, word2_key
    from src.blocking_embedding import build_candidates_embedding
    from src.blocking_tfidf import build_candidates_tfidf
    from src.data_loader import (
        cached_normalize_address,
        cached_normalize_country,
        cached_normalize_name,
        clear_norm_caches,
        load_ground_truth,
    )


def _merge_into(target: Dict[str, List[str]], source: Dict[str, List[str]]) -> None:
    """In-place merge of candidate dictionaries while preserving ranking order."""
    for s1_id, cids in source.items():
        if s1_id not in target:
            target[s1_id] = list(cids)
        else:
            seen = set(target[s1_id])
            for cid in cids:
                if cid not in seen:
                    seen.add(cid)
                    target[s1_id].append(cid)


def union_candidates(*cand_dicts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Union candidates across multiple dictionary outputs."""
    out: Dict[str, List[str]] = {}
    for d in cand_dicts:
        _merge_into(out, d)
    return out


def compute_frame_keys(df: pd.DataFrame, prune_raw: bool = True) -> pd.DataFrame:
    """Compute normalized text representations and 4 inverted-index blocking keys."""
    df = df.copy()
    raw_names = df["business_name"].fillna("").astype(str).tolist()
    raw_addrs = df["business_address"].fillna("").astype(str).tolist()
    raw_countries = df["country"].fillna("").astype(str).tolist()

    norm_names = [cached_normalize_name(n) for n in raw_names]
    norm_addrs = [cached_normalize_address(a) for a in raw_addrs]
    norm_countries = [
        config.normalize_country_key(cached_normalize_country(c))
        for c in raw_countries
    ]

    df["name_norm"] = norm_names
    df["addr_norm"] = norm_addrs
    df["country_key"] = norm_countries
    df["match_text"] = [match_text(n, a) for n, a in zip(norm_names, norm_addrs)]

    # 4 distinct blocking keys for multi-view candidate retrieval
    df["k1"] = [f"{c}\x1f{name_prefix(n, 3)}" for c, n in zip(norm_countries, norm_names)]
    df["k2"] = [f"{c}\x1fstr_{stripped_prefix(n, 3)}" for c, n in zip(norm_countries, norm_names)]
    df["k3"] = [f"{c}\x1faddr_{addr_key(a)}" for c, a in zip(norm_countries, norm_addrs)]
    df["k4"] = [f"{c}\x1fw2_{word2_key(n, 3)}" for c, n in zip(norm_countries, norm_names)]

    # Prune unneeded raw text columns to conserve memory
    if prune_raw:
        for col in ("business_name", "business_address", "country"):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    return df


def add_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add blocking keys to already normalized dataframe."""
    norm_names = df["name_norm"].fillna("").tolist()
    norm_addrs = df["addr_norm"].fillna("").tolist()
    norm_countries = df["country_key"].fillna("").tolist()

    df["k1"] = [f"{c}\x1f{name_prefix(n, 3)}" for c, n in zip(norm_countries, norm_names)]
    df["k2"] = [f"{c}\x1fstr_{stripped_prefix(n, 3)}" for c, n in zip(norm_countries, norm_names)]
    df["k3"] = [f"{c}\x1faddr_{addr_key(a)}" for c, a in zip(norm_countries, norm_addrs)]
    df["k4"] = [f"{c}\x1fw2_{word2_key(n, 3)}" for c, n in zip(norm_countries, norm_names)]
    return df


def run_4key_blocking(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    top_k_k1: int = 60,
    top_k_k2: int = 30,
    top_k_k3: int = 30,
    top_k_k4: int = 20,
    fallback_min_cands: int = 5,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Run candidate retrieval sequentially across 4 keys with memory cleanup between passes."""
    t0 = time.perf_counter()
    cands: Dict[str, List[str]] = {}

    # Key 1: 3-character prefix index
    s1["block_key"] = s1["k1"]
    s23["block_key"] = s23["k1"]
    if verbose:
        print(f"    [K1 TF-IDF] top_k={top_k_k1}...")
    c1 = build_candidates_tfidf(s1, s23, top_k=top_k_k1, verbose=False)
    _merge_into(cands, c1)
    del c1
    gc.collect()

    # Key 2: Stripped noise prefix
    s1["block_key"] = s1["k2"]
    s23["block_key"] = s23["k2"]
    if verbose:
        print(f"    [K2 TF-IDF] top_k={top_k_k2}...")
    c2 = build_candidates_tfidf(s1, s23, top_k=top_k_k2, verbose=False)
    _merge_into(cands, c2)
    del c2
    gc.collect()

    # Key 3: Address composite key
    s1["block_key"] = s1["k3"]
    s23["block_key"] = s23["k3"]
    if verbose:
        print(f"    [K3 TF-IDF] top_k={top_k_k3}...")
    c3 = build_candidates_tfidf(s1, s23, top_k=top_k_k3, verbose=False)
    _merge_into(cands, c3)
    del c3
    gc.collect()

    # Key 4: Second-word prefix
    s1["block_key"] = s1["k4"]
    s23["block_key"] = s23["k4"]
    if verbose:
        print(f"    [K4 TF-IDF] top_k={top_k_k4}...")
    c4 = build_candidates_tfidf(s1, s23, top_k=top_k_k4, verbose=False)
    _merge_into(cands, c4)
    del c4
    gc.collect()

    # Targeted semantic fallback for difficult queries with very few lexical candidates
    all_s1_ids = set(s1["entity_id"])
    low_cand_ids = [eid for eid in all_s1_ids if len(cands.get(eid, [])) < fallback_min_cands]

    if low_cand_ids:
        if verbose:
            print(f"    [Fallback MiniLM] {len(low_cand_ids)} queries with < {fallback_min_cands} candidates...")
        s1_fallback = s1[s1["entity_id"].isin(set(low_cand_ids))].copy()
        s1_fallback["block_key"] = s1_fallback["k1"]
        s23["block_key"] = s23["k1"]
        c_fb = build_candidates_embedding(s1_fallback, s23, top_k=30, verbose=False)
        _merge_into(cands, c_fb)
        del c_fb, s1_fallback
        gc.collect()

    # Ensure every query ID is represented in output mapping
    for eid in all_s1_ids:
        if eid not in cands:
            cands[eid] = []

    if verbose:
        dt = time.perf_counter() - t0
        print(f"    Blocked {len(s1):,} queries against {len(s23):,} candidate records in {dt:.1f}s")
    return cands


def run_blocking(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    top_k: int = 30,
    signals: Sequence[str] = ("tfidf", "embedding"),
    verbose: bool = False,
) -> Dict[str, List[str]]:
    """Entry point for blocking."""
    if "k1" not in s1.columns:
        s1 = compute_frame_keys(s1)
    if "k1" not in s23.columns:
        s23 = compute_frame_keys(s23)
    return run_4key_blocking(s1, s23, verbose=verbose)


def run_full_dataset_pipeline(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    out_candidate_tsv: Optional[Path] = None,
    ground_truth_path: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], Optional[Dict[str, object]], float]:
    """Execute blocking partition by partition across entire dataset."""
    t_global = time.perf_counter()

    if verbose:
        print("=" * 70)
        print(f"BLOCKING PIPELINE: {s1_path.name}")
        print("=" * 70)

    s1_pl = pl.read_csv(
        s1_path,
        separator=config.SEP,
        columns=["entity_id", "business_name", "business_address", "country"],
        truncate_ragged_lines=True,
    )
    s1_pl = s1_pl.with_columns(pl.col("country").str.to_lowercase().alias("country_key"))
    s1_all_ids = s1_pl["entity_id"].to_list()
    s1_countries = s1_pl["country_key"].to_list()
    distinct_countries = sorted(list(set(s1_countries)))
    if verbose:
        print(f"Source 1 total entities: {len(s1_all_ids):,} across countries: {distinct_countries}")

    all_candidates: Dict[str, List[str]] = {}

    out_f = None
    if out_candidate_tsv is not None:
        out_candidate_tsv.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(out_candidate_tsv, "w", encoding="utf-8")
        out_f.write(f"{config.CANDIDATE_HEADER[0]}{config.SEP}{config.CANDIDATE_HEADER[1]}\n")

    for ck in distinct_countries:
        if verbose:
            print(f"\nProcessing Country: {ck.upper()}")

        s1_c_pl = s1_pl.filter(pl.col("country_key") == ck)
        s1_c_df = compute_frame_keys(s1_c_pl.to_pandas(), prune_raw=True)
        del s1_c_pl
        gc.collect()

        if len(s1_c_df) == 0:
            continue

        s23_dfs = []
        for src_path in (s2_path, s3_path):
            src_pl = pl.read_csv(
                src_path,
                separator=config.SEP,
                columns=["entity_id", "business_name", "business_address", "country"],
                truncate_ragged_lines=True,
            )
            hit = src_pl.filter(pl.col("country").str.to_lowercase() == ck)
            if len(hit) > 0:
                s23_dfs.append(hit.to_pandas())
            del src_pl, hit
            gc.collect()

        if s23_dfs:
            s23_c_df = pd.concat(s23_dfs, ignore_index=True)
            del s23_dfs
            s23_c_df = compute_frame_keys(s23_c_df, prune_raw=True)
        else:
            s23_c_df = pd.DataFrame(columns=["entity_id", "name_norm", "addr_norm", "country_key", "match_text", "k1", "k2", "k3", "k4"])

        cands_c = run_4key_blocking(s1_c_df, s23_c_df, verbose=verbose)
        all_candidates.update(cands_c)

        if out_f is not None:
            for sid, c_list in cands_c.items():
                out_f.write(f"{sid}{config.SEP}{','.join(c_list)}\n")
            out_f.flush()

        del s1_c_df, s23_c_df, cands_c
        gc.collect()

        clear_norm_caches()
        gc.collect()

    if out_f is not None:
        out_f.close()
        if verbose:
            print(f"\nCandidate pairs written to {out_candidate_tsv}")

    total_time = time.perf_counter() - t_global

    # Evaluate candidate recall if ground truth is supplied
    report = None
    if ground_truth_path is not None:
        if verbose:
            print("\nEvaluating recall against ground truth...")
        gt = load_ground_truth(ground_truth_path)
        s1_meta = pd.DataFrame({
            "entity_id": s1_all_ids,
            "country_key": s1_countries,
        })
        report = evaluate_recall(all_candidates, gt, s1_meta)
        if verbose:
            print_recall_report(report)

    if verbose:
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print("=" * 70)

    return all_candidates, report, total_time


def candidates_to_frame(cands: Dict[str, List[str]],
                        all_s1_ids: Iterable[str]) -> pd.DataFrame:
    rows = [
        {"source1_entity_id": sid, "candidate_entity_ids": ",".join(cands.get(sid, []))}
        for sid in all_s1_ids
    ]
    return pd.DataFrame(rows, columns=config.CANDIDATE_HEADER)


def write_candidate_pairs(cands: Dict[str, List[str]],
                          all_s1_ids: Iterable[str],
                          path: str | Path) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = candidates_to_frame(cands, all_s1_ids)
    frame.to_csv(path, sep=config.SEP, index=False, encoding="utf-8")
    return frame


def write_candidate_pairs_streaming(cands: Dict[str, List[str]],
                                    all_s1_ids: Iterable[str],
                                    path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{config.CANDIDATE_HEADER[0]}{config.SEP}{config.CANDIDATE_HEADER[1]}\n")
        for sid in all_s1_ids:
            c_list = cands.get(sid, [])
            f.write(f"{sid}{config.SEP}{','.join(c_list)}\n")
            n += 1
    return n


def evaluate_recall(cands: Dict[str, List[str]], gt: pd.DataFrame,
                    s1: pd.DataFrame) -> Dict[str, object]:
    """Calculate recall ceiling against labeled ground truth."""
    country_of = dict(zip(s1["entity_id"], s1["country_key"]))
    total_links = retrieved_links = 0
    per_country: Dict[str, List[int]] = {}
    entities_full = entities_scored = 0
    n_cands: List[int] = []
    singleton_with_cands = singletons = 0

    for _, row in gt.iterrows():
        sid = row["source1_entity_id"]
        truth = row["matched_list"] if isinstance(row["matched_list"], list) else []
        got = set(cands.get(sid, []))
        n_cands.append(len(got))
        if not truth:
            singletons += 1
            if got:
                singleton_with_cands += 1
            continue
        entities_scored += 1
        hit = sum(1 for t in truth if t in got)
        total_links += len(truth)
        retrieved_links += hit
        if hit == len(truth):
            entities_full += 1
        ck = country_of.get(sid, "__unknown__")
        acc = per_country.setdefault(ck, [0, 0])
        acc[0] += hit
        acc[1] += len(truth)

    counts = np.asarray(n_cands, dtype=np.int64)
    report: Dict[str, object] = {
        "n_entities": len(gt),
        "overall_link_recall": (retrieved_links / total_links) if total_links else 1.0,
        "retrieved_links": retrieved_links,
        "total_links": total_links,
        "per_country_recall": {
            ck: (r / t if t else 1.0) for ck, (r, t) in sorted(per_country.items())
        },
        "entities_full_recall_rate": (entities_full / entities_scored) if entities_scored else 1.0,
        "entities_full_recall": entities_full,
        "entities_with_matches": entities_scored,
        "singletons": singletons,
        "singleton_with_candidates_rate": (
            singleton_with_cands / singletons if singletons else 0.0
        ),
        "mean_candidates": float(counts.mean()) if len(counts) else 0.0,
        "median_candidates": float(np.median(counts)) if len(counts) else 0.0,
        "max_candidates": int(counts.max()) if len(counts) else 0,
        "empty_candidate_entities": int((counts == 0).sum()),
    }
    return report


def print_recall_report(report: Dict[str, object]) -> None:
    print("\n===== RECALL SUMMARY (candidates vs. ground truth) =====")
    print(f"  entities scored            : {report['n_entities']:,}")
    print(f"  OVERALL link recall        : {report['overall_link_recall']:.4f} "
          f"({report['retrieved_links']:,}/{report['total_links']:,})")
    print("  per-country link recall    :")
    for ck, r in report["per_country_recall"].items():
        print(f"      {ck:<12} {r:.4f}")
    print(f"  entities w/ FULL recall    : {report['entities_full_recall_rate']:.4f} "
          f"({report['entities_full_recall']:,}/{report['entities_with_matches']:,})")
    print(f"  singletons w/ candidates   : {report['singleton_with_candidates_rate']:.4f}")
    print(f"  candidates/entity mean/med : {report['mean_candidates']:.1f} / "
          f"{report['median_candidates']:.0f}  (max {report['max_candidates']}, "
          f"{report['empty_candidate_entities']} empty)")
    print("========================================================\n")


if __name__ == "__main__":
    import argparse

    config.enable_utf8_stdout()

    parser = argparse.ArgumentParser(description="Blocking pipeline")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--out", default=str(config.CANDIDATE_PAIRS_PATH))
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        print("\nRunning on train dataset...")
        run_full_dataset_pipeline(
            s1_path=config.TRAIN_SOURCE1,
            s2_path=config.TRAIN_SOURCE2,
            s3_path=config.TRAIN_SOURCE3,
            ground_truth_path=config.TRAIN_GROUND_TRUTH,
            verbose=True,
        )

    if args.mode in ("test", "both"):
        print("\nRunning on test dataset...")
        run_full_dataset_pipeline(
            s1_path=config.TEST_SOURCE1,
            s2_path=config.TEST_SOURCE2,
            s3_path=config.TEST_SOURCE3,
            out_candidate_tsv=Path(args.out),
            verbose=True,
        )
