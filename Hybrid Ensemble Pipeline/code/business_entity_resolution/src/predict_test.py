from __future__ import annotations

import ctypes
import gc
import json
import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import polars as pl
import psutil
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import HashingVectorizer

try:
    from . import config
    from .block_keys import addr_key, name_prefix, stripped_prefix, word2_key
    from .features import fast_soundex, token_jaccard
    from .text_normalize import normalize_address, normalize_country, normalize_name
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.block_keys import addr_key, name_prefix, stripped_prefix, word2_key
    from src.features import fast_soundex, token_jaccard
    from src.text_normalize import normalize_address, normalize_country, normalize_name


def trim_process_memory():
    """Trigger garbage collection and release unused pages back to the OS."""
    gc.collect()
    if sys.platform == "win32":
        try:
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            pass


GLOBAL_VEC = HashingVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    n_features=2048,
    alternate_sign=False,
    norm="l2",
    dtype=np.float32,
)


def load_model_and_meta() -> Tuple[object, Dict[str, float]]:
    """Load pre-trained matcher model and country decision thresholds."""
    model_path = config.MODELS_DIR / "lgbm_matcher.joblib"
    meta_path = config.MODELS_DIR / "model_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

    model = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    thresholds = meta.get("thresholds", {"india": 0.80, "us": 0.80, "france": 0.80})
    return model, thresholds


def run_test_pipeline(
    chunk_size: int = 2000,
    resume: bool = True,
    target_countries: Optional[List[str]] = None,
):
    """End-to-end streaming inference pipeline."""
    config.enable_utf8_stdout()
    config.ensure_output_dirs()
    proc = psutil.Process()

    print("=" * 70, flush=True)
    print("TEST PREDICTION & SUBMISSION GENERATION", flush=True)
    print("=" * 70, flush=True)

    model, thresholds = load_model_and_meta()
    print(f"Loaded trained model: {type(model).__name__}", flush=True)
    print(f"Per-country thresholds: {thresholds}", flush=True)

    matching_out = config.MATCHING_RESULTS_PATH
    candidate_out = config.CANDIDATE_PAIRS_PATH

    processed_s1_ids: Set[str] = set()
    claimed_candidates: Set[str] = set()

    # Check for existing partial output to resume safely
    if resume and matching_out.exists() and os.path.getsize(matching_out) > 0:
        with open(matching_out, "r", encoding="utf-8") as f_match:
            for line in f_match:
                parts = line.strip().split(config.SEP)
                if parts and parts[0] != config.MATCHING_HEADER[0]:
                    processed_s1_ids.add(parts[0])
                    if len(parts) > 1 and parts[1]:
                        for cand in parts[1].split(","):
                            cand = cand.strip()
                            if cand:
                                claimed_candidates.add(cand)
        print(f"Resuming execution: Found {len(processed_s1_ids):,} already processed entities.", flush=True)
        print(f"Loaded {len(claimed_candidates):,} claimed candidates.", flush=True)
    else:
        with open(matching_out, "w", encoding="utf-8") as f_match:
            f_match.write(f"{config.MATCHING_HEADER[0]}{config.SEP}{config.MATCHING_HEADER[1]}\n")

        with open(candidate_out, "w", encoding="utf-8") as f_cand:
            f_cand.write(f"{config.CANDIDATE_HEADER[0]}{config.SEP}{config.CANDIDATE_HEADER[1]}\n")

    t0_global = time.perf_counter()
    TOTAL_TEST_S1 = 1_732_544
    countries = target_countries or ["france", "us", "india"]
    total_s1_processed = len(processed_s1_ids)

    # Process one country partition at a time to control peak RAM
    for c in countries:
        th = thresholds.get(c, 0.80)
        print(f"\nProcessing Country: {c.upper()} (Threshold = {th:.2f})", flush=True)
        t_country = time.perf_counter()

        print(f"  Scanning Source 1 for {c.upper()}...", flush=True)
        s1_c_df = (
            pl.scan_csv(
                config.TEST_SOURCE1,
                separator=config.SEP,
                truncate_ragged_lines=True,
            )
            .select(["entity_id", "business_name", "business_address", "country"])
            .filter(pl.col("country").fill_null("").str.strip_chars().str.to_lowercase() == c)
            .collect()
            .to_pandas()
        )
        n_s1_c = len(s1_c_df)
        print(f"  Source 1 {c.upper()} count: {n_s1_c:,} (RAM: {proc.memory_info().rss / (1024*1024):.1f} MB)", flush=True)

        # Skip already completed records when resuming
        if resume and len(processed_s1_ids) > 0:
            s1_c_ids = set(s1_c_df["entity_id"])
            remaining_ids = s1_c_ids - processed_s1_ids
            if len(remaining_ids) == 0:
                print(f"  All {n_s1_c:,} entities for {c.upper()} already processed. Skipping...", flush=True)
                del s1_c_df
                gc.collect()
                continue
            elif len(remaining_ids) < len(s1_c_ids):
                print(f"  Resuming {c.upper()}: {len(remaining_ids):,} of {n_s1_c:,} entities remaining.", flush=True)
                s1_c_df = s1_c_df[s1_c_df["entity_id"].isin(remaining_ids)].reset_index(drop=True)
                n_s1_c = len(s1_c_df)

        print(f"  Scanning Source 2 & 3 for {c.upper()}...", flush=True)
        s23_dfs = []
        for p in (config.TEST_SOURCE2, config.TEST_SOURCE3):
            hit = pl.scan_csv(
                p,
                separator=config.SEP,
                truncate_ragged_lines=True,
            ).select(["entity_id", "business_name", "business_address", "country"]).filter(
                pl.col("country").fill_null("").str.strip_chars().str.to_lowercase() == c
            ).collect()
            if len(hit):
                s23_dfs.append(hit.to_pandas())
            del hit
            gc.collect()

        if s23_dfs:
            s23_c_df = pd.concat(s23_dfs, ignore_index=True).drop_duplicates(subset=["entity_id"])
            del s23_dfs
            gc.collect()
        else:
            s23_c_df = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

        print(f"  Candidate pool {c.upper()} size: {len(s23_c_df):,} records", flush=True)

        if len(s23_c_df) == 0:
            with open(matching_out, "a", encoding="utf-8") as f_match, open(candidate_out, "a", encoding="utf-8") as f_cand:
                for eid in s1_c_df["entity_id"]:
                    f_match.write(f"{eid}{config.SEP}\n")
                    f_cand.write(f"{eid}{config.SEP}\n")
            total_s1_processed += n_s1_c
            del s1_c_df, s23_c_df
            gc.collect()
            continue

        # Build in-memory inverted indices for candidate pool
        s23_ids = s23_c_df["entity_id"].tolist()
        raw_names = s23_c_df["business_name"].fillna("").tolist()
        raw_addrs = s23_c_df["business_address"].fillna("").tolist()
        del s23_c_df
        trim_process_memory()

        k1_index: Dict[str, List[int]] = defaultdict(list)
        k2_index: Dict[str, List[int]] = defaultdict(list)
        k3_index: Dict[str, List[int]] = defaultdict(list)

        s23_norm_names: List[str] = [None] * len(s23_ids)
        s23_norm_addrs: List[str] = [None] * len(s23_ids)

        for idx in range(len(s23_ids)):
            n = normalize_name(str(raw_names[idx]))
            a = normalize_address(str(raw_addrs[idx]))
            raw_names[idx] = None
            raw_addrs[idx] = None
            s23_norm_names[idx] = n
            s23_norm_addrs[idx] = a

            k1 = name_prefix(n, 3)
            if k1:
                k1_index[k1].append(idx)
            k2 = stripped_prefix(n, 3)
            if k2:
                k2_index[k2].append(idx)
            k3 = addr_key(a)
            if k3 != "__none__":
                k3_index[k3].append(idx)

        del raw_names, raw_addrs
        trim_process_memory()

        print(f"  Indexed blocks for {c.upper()} (RAM: {proc.memory_info().rss / (1024*1024):.1f} MB)", flush=True)

        s1_ids = s1_c_df["entity_id"].tolist()
        raw_s1_names = s1_c_df["business_name"].fillna("").tolist()
        raw_s1_addrs = s1_c_df["business_address"].fillna("").tolist()
        del s1_c_df
        trim_process_memory()

        s1_norm_names: List[str] = [None] * len(s1_ids)
        s1_norm_addrs: List[str] = [None] * len(s1_ids)
        for i in range(len(s1_ids)):
            s1_norm_names[i] = normalize_name(str(raw_s1_names[i]))
            raw_s1_names[i] = None
            s1_norm_addrs[i] = normalize_address(str(raw_s1_addrs[i]))
            raw_s1_addrs[i] = None
        del raw_s1_names, raw_s1_addrs
        trim_process_memory()

        n_chunks = (n_s1_c + chunk_size - 1) // chunk_size

        # Stream predictions in micro-chunks of 2,000 entities
        for ch_idx in range(n_chunks):
            start = ch_idx * chunk_size
            end = min(start + chunk_size, n_s1_c)

            ch_pairs_s1: List[str] = []
            ch_pairs_cand: List[str] = []
            pair_s1_local_idx: List[int] = []
            pair_cand_pool_idx: List[int] = []

            chunk_cands_map: Dict[str, List[str]] = {}

            # Generate candidate sets from index lookups
            for local_i, i in enumerate(range(start, end)):
                sid = s1_ids[i]
                n = s1_norm_names[i]
                a = s1_norm_addrs[i]
                k1_val = name_prefix(n, 3)
                k2_val = stripped_prefix(n, 3)
                k3_val = addr_key(a)

                cand_idx_set: Set[int] = set()
                if k1_val and k1_val in k1_index:
                    cand_idx_set.update(k1_index[k1_val][:20])
                if k2_val and k2_val in k2_index:
                    cand_idx_set.update(k2_index[k2_val][:10])
                if k3_val != "__none__" and k3_val in k3_index:
                    cand_idx_set.update(k3_index[k3_val][:10])

                c_list = [s23_ids[idx] for idx in cand_idx_set]
                chunk_cands_map[sid] = c_list

                for idx in cand_idx_set:
                    ch_pairs_s1.append(sid)
                    ch_pairs_cand.append(s23_ids[idx])
                    pair_s1_local_idx.append(local_i)
                    pair_cand_pool_idx.append(idx)

            chunk_matches_map: Dict[str, List[str]] = defaultdict(list)
            n_pairs = len(ch_pairs_s1)

            # Featurize and score candidate pairs
            if n_pairs > 0:
                unique_cand_indices = sorted(list(set(pair_cand_pool_idx)))
                cand_pool_to_offset = {idx: o for o, idx in enumerate(unique_cand_indices)}

                s1_chunk_texts = [f"{s1_norm_names[i]} {s1_norm_addrs[i]}" for i in range(start, end)]
                cand_chunk_texts = [f"{s23_norm_names[idx]} {s23_norm_addrs[idx]}" for idx in unique_cand_indices]

                m_s1 = GLOBAL_VEC.transform(s1_chunk_texts)
                m_cand = GLOBAL_VEC.transform(cand_chunk_texts)

                s1_offsets = pair_s1_local_idx
                cand_offsets = [cand_pool_to_offset[idx] for idx in pair_cand_pool_idx]

                f_tfidf = np.empty(n_pairs, dtype=np.float32)
                batch_sz = 5000
                for b_start in range(0, n_pairs, batch_sz):
                    b_end = min(b_start + batch_sz, n_pairs)
                    b_s1 = m_s1[s1_offsets[b_start:b_end]]
                    b_cand = m_cand[cand_offsets[b_start:b_end]]
                    f_tfidf[b_start:b_end] = np.asarray(b_s1.multiply(b_cand).sum(axis=1), dtype=np.float32).ravel()

                s1_n_toks = [s1_norm_names[i].split() for i in range(start, end)]
                s1_a_toks = [s1_norm_addrs[i].split() for i in range(start, end)]
                s1_snd = [fast_soundex(s1_norm_names[i]) for i in range(start, end)]

                cand_n_toks = [s23_norm_names[idx].split() for idx in unique_cand_indices]
                cand_a_toks = [s23_norm_addrs[idx].split() for idx in unique_cand_indices]
                cand_snd = [fast_soundex(s23_norm_names[idx]) for idx in unique_cand_indices]

                f_lev = np.zeros(n_pairs, dtype=np.float32)
                f_jw = np.zeros(n_pairs, dtype=np.float32)
                f_name_jaccard = np.zeros(n_pairs, dtype=np.float32)
                f_addr_jaccard = np.zeros(n_pairs, dtype=np.float32)
                f_phonetic = np.zeros(n_pairs, dtype=np.int8)
                f_country = np.ones(n_pairs, dtype=np.int8)
                f_len_diff = np.zeros(n_pairs, dtype=np.float32)

                for p in range(n_pairs):
                    s_off = s1_offsets[p]
                    c_off = cand_offsets[p]
                    na = s1_norm_names[start + s_off]
                    nb = s23_norm_names[unique_cand_indices[c_off]]
                    f_lev[p] = Levenshtein.normalized_similarity(na, nb)
                    f_jw[p] = JaroWinkler.similarity(na, nb)
                    f_name_jaccard[p] = token_jaccard(s1_n_toks[s_off], cand_n_toks[c_off])
                    f_addr_jaccard[p] = token_jaccard(s1_a_toks[s_off], cand_a_toks[c_off])
                    f_phonetic[p] = 1 if s1_snd[s_off] == cand_snd[c_off] else 0
                    max_l = max(len(na), len(nb), 1)
                    f_len_diff[p] = abs(len(na) - len(nb)) / max_l

                X_chunk = np.column_stack([
                    f_lev, f_jw, f_name_jaccard, f_addr_jaccard,
                    f_tfidf, f_phonetic, f_country, f_len_diff
                ])
                probs = model.predict_proba(X_chunk)[:, 1]

                # Resolve 1-to-1 graph consistency greedily by descending probability
                sorted_idx = np.argsort(-probs)
                for idx in sorted_idx:
                    prob = float(probs[idx])
                    if prob < th:
                        break
                    sid = ch_pairs_s1[idx]
                    cid = ch_pairs_cand[idx]
                    if cid not in claimed_candidates:
                        claimed_candidates.add(cid)
                        chunk_matches_map[sid].append(cid)

                del m_s1, m_cand, f_tfidf, X_chunk, probs, sorted_idx
                del s1_n_toks, s1_a_toks, s1_snd, cand_n_toks, cand_a_toks, cand_snd
                del f_lev, f_jw, f_name_jaccard, f_addr_jaccard, f_phonetic, f_country, f_len_diff

            # Append results directly to output TSVs
            with open(matching_out, "a", encoding="utf-8") as f_match:
                for i in range(start, end):
                    sid = s1_ids[i]
                    matches = chunk_matches_map.get(sid, [])
                    match_str = ",".join(matches)
                    f_match.write(f"{sid}{config.SEP}{match_str}\n")

            with open(candidate_out, "a", encoding="utf-8") as f_cand:
                for i in range(start, end):
                    sid = s1_ids[i]
                    cands = chunk_cands_map.get(sid, [])
                    matches = chunk_matches_map.get(sid, [])
                    if matches:
                        cand_set = set(cands)
                        for m in matches:
                            if m not in cand_set:
                                cands.append(m)
                                cand_set.add(m)
                    cand_str = ",".join(cands)
                    f_cand.write(f"{sid}{config.SEP}{cand_str}\n")

            total_s1_processed += (end - start)
            del ch_pairs_s1, ch_pairs_cand, pair_s1_local_idx, pair_cand_pool_idx
            del chunk_cands_map, chunk_matches_map

            if (ch_idx + 1) % 5 == 0 or (ch_idx + 1) == n_chunks:
                trim_process_memory()
                ram_mb = proc.memory_info().rss / (1024 * 1024)
                print(
                    f"    Chunk {ch_idx + 1}/{n_chunks} ({total_s1_processed:,}/{TOTAL_TEST_S1:,} total) | RAM: {ram_mb:.1f} MB",
                    flush=True,
                )

        dt_c = time.perf_counter() - t_country
        print(f"  Finished country {c.upper()} in {dt_c:.1f}s", flush=True)

        del s23_ids, k1_index, k2_index, k3_index
        del s1_norm_names, s1_norm_addrs, s23_norm_names, s23_norm_addrs
        del s1_ids
        trim_process_memory()

    dt_total = time.perf_counter() - t0_global
    print("\n" + "=" * 70, flush=True)
    print(f"Test prediction completed in {dt_total:.1f}s ({dt_total/60:.2f} min)", flush=True)
    print(f"Matching Results : {matching_out} ({os.path.getsize(matching_out):,} bytes)", flush=True)
    print(f"Candidate Pairs  : {candidate_out} ({os.path.getsize(candidate_out):,} bytes)", flush=True)
    print("=" * 70, flush=True)

    if total_s1_processed >= TOTAL_TEST_S1:
        print("\n--- Running Submission Validator ---", flush=True)
        val_cmd = [
            sys.executable,
            str(config.PROJECT_ROOT / "utils" / "validate_submission.py"),
            "--matching", str(matching_out),
            "--candidate", str(candidate_out),
            "--test-dir", str(config.TEST_DIR),
        ]
        res = subprocess.run(val_cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Prediction Pipeline")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Batch size for entities")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh and overwrite existing results")
    parser.add_argument("--countries", type=str, default=None, help="Comma-separated countries to run")
    cli_args = parser.parse_args()

    c_list = [c.strip().lower() for c in cli_args.countries.split(",")] if cli_args.countries else None
    run_test_pipeline(
        chunk_size=cli_args.chunk_size,
        resume=not cli_args.no_resume,
        target_countries=c_list,
    )
