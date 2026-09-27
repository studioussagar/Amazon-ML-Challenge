from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from . import config
from .features import compute_features_batch
from .text_normalize import normalize_address, normalize_country, normalize_name

FEATURE_COLS = [
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_jaccard",
    "addr_token_jaccard",
    "tfidf_cosine",
    "phonetic_match",
    "country_match",
    "length_diff",
]


def compute_entity_f05(pred_set: Set[str], true_set: Set[str]) -> float:
    """Compute F_0.5 score for a single entity (beta=0.5 penalizes false positives heavily)."""
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    return float((1.25 * precision * recall) / (0.25 * precision + recall))


def evaluate_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    entity_countries: Dict[str, str],
) -> Tuple[float, Dict[str, float]]:
    """Compute Macro F_0.5 across all entities and breakdown by country."""
    all_scores: List[float] = []
    country_scores: Dict[str, List[float]] = {}

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(pred_set, true_set)
        all_scores.append(score)

        c = entity_countries.get(s1_id, "unknown")
        if c not in country_scores:
            country_scores[c] = []
        country_scores[c].append(score)

    overall_macro = float(np.mean(all_scores)) if all_scores else 0.0
    per_country = {c: float(np.mean(scs)) for c, scs in country_scores.items()}
    return overall_macro, per_country


def build_training_features(
    candidate_file: Path,
    sample_s1_df: pd.DataFrame,
    sample_pool_df: pd.DataFrame,
    ground_truth_map: Dict[str, Set[str]],
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Extract features in batches for candidate pairs."""
    feat_parquet = config.INTERIM_DIR / "sample_features.parquet"
    if feat_parquet.exists():
        print(f"Loading cached features from {feat_parquet}...", flush=True)
        all_df = pd.read_parquet(feat_parquet)
        print(f"Features shape: {all_df.shape} (Positives: {all_df['label'].sum():,})", flush=True)
        return all_df

    print("Building lookup dictionaries for entities...", flush=True)
    s1_dict = sample_s1_df.set_index("entity_id")[["name_norm", "addr_norm", "country_key"]].to_dict("index")
    pool_dict = sample_pool_df.set_index("entity_id")[["name_norm", "addr_norm", "country_key"]].to_dict("index")

    pairs: List[Tuple[str, str]] = []
    with open(candidate_file, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            s1_id, cand_str = parts[0], parts[1]
            if not cand_str:
                continue
            cands = cand_str.split(",")
            for cid in cands:
                if cid:
                    pairs.append((s1_id, cid))

    print(f"Total candidate pairs to featurize: {len(pairs):,}", flush=True)

    chunks_features: List[pd.DataFrame] = []
    n_pairs = len(pairs)

    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        chunk_pairs = pairs[start:end]

        s1_names, s1_addrs, s1_countries = [], [], []
        cand_names, cand_addrs, cand_countries = [], [], []
        labels = []
        s1_ids, cand_ids = [], []

        for s1_id, cid in chunk_pairs:
            s1_info = s1_dict.get(s1_id)
            cand_info = pool_dict.get(cid)
            if not s1_info or not cand_info:
                continue

            s1_ids.append(s1_id)
            cand_ids.append(cid)
            s1_names.append(s1_info["name_norm"])
            s1_addrs.append(s1_info["addr_norm"])
            s1_countries.append(s1_info["country_key"])

            cand_names.append(cand_info["name_norm"])
            cand_addrs.append(cand_info["addr_norm"])
            cand_countries.append(cand_info["country_key"])

            is_match = 1 if cid in ground_truth_map.get(s1_id, set()) else 0
            labels.append(is_match)

        if not s1_ids:
            continue

        f_dict = compute_features_batch(
            s1_names, s1_addrs, s1_countries,
            cand_names, cand_addrs, cand_countries,
        )
        f_dict["source1_entity_id"] = s1_ids
        f_dict["candidate_id"] = cand_ids
        f_dict["label"] = np.asarray(labels, dtype=np.int8)

        df_chunk = pd.DataFrame(f_dict)
        chunks_features.append(df_chunk)

        del s1_names, s1_addrs, s1_countries, cand_names, cand_addrs, cand_countries, labels, s1_ids, cand_ids, f_dict
        gc.collect()

    all_df = pd.concat(chunks_features, ignore_index=True)
    del chunks_features
    gc.collect()

    all_df.to_parquet(feat_parquet, index=False)
    print(f"Saved features -> {feat_parquet}", flush=True)
    print(f"Features shape: {all_df.shape} (Positives: {all_df['label'].sum():,})", flush=True)
    return all_df


def train_matching_model():
    """Train LightGBM binary classifier and sweep per-country decision thresholds."""
    config.ensure_output_dirs()

    print("Loading ground truth...", flush=True)
    gt_pl = pl.read_csv(
        config.TRAIN_GROUND_TRUTH,
        separator=config.SEP,
        columns=["source1_entity_id", "matched_entity_ids"],
        truncate_ragged_lines=True,
    )
    gt_map: Dict[str, Set[str]] = {}
    for row in gt_pl.iter_rows(named=True):
        sid = row["source1_entity_id"]
        mids = str(row["matched_entity_ids"] or "").strip()
        gt_map[sid] = set(mids.split(",")) if mids else set()
    del gt_pl
    gc.collect()

    print("Loading sample entities...", flush=True)
    s1_pl = pl.read_csv(
        config.TRAIN_SOURCE1,
        separator=config.SEP,
        columns=["entity_id", "business_name", "business_address", "country"],
        truncate_ragged_lines=True,
    )
    s1_df = s1_pl.to_pandas()
    del s1_pl
    gc.collect()

    cand_path = config.INTERIM_DIR / "sample_candidate_pairs.tsv"
    sample_s1_ids = set()
    with open(cand_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            sid = line.split("\t")[0].strip()
            if sid:
                sample_s1_ids.add(sid)

    s1_sample_df = s1_df[s1_df["entity_id"].isin(sample_s1_ids)].copy()
    del s1_df
    gc.collect()

    s1_sample_df["name_norm"] = [normalize_name(n) for n in s1_sample_df["business_name"]]
    s1_sample_df["addr_norm"] = [normalize_address(a) for a in s1_sample_df["business_address"]]
    s1_sample_df["country_key"] = [config.normalize_country_key(normalize_country(c)) for c in s1_sample_df["country"]]

    print("Loading candidate records from Source 2 & 3...", flush=True)
    cand_ids_needed = set()
    with open(cand_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1]:
                cand_ids_needed.update(parts[1].split(","))

    cand_ids_list = list(cand_ids_needed)
    pool_dfs = []
    for p in (config.TRAIN_SOURCE2, config.TRAIN_SOURCE3):
        hit = pl.scan_csv(
            p,
            separator=config.SEP,
            truncate_ragged_lines=True,
        ).select(["entity_id", "business_name", "business_address", "country"]).filter(
            pl.col("entity_id").is_in(cand_ids_list)
        ).collect()
        if len(hit):
            pool_dfs.append(hit.to_pandas())
        del hit
        gc.collect()

    pool_df = pd.concat(pool_dfs, ignore_index=True).drop_duplicates(subset=["entity_id"])
    del pool_dfs
    gc.collect()

    pool_df["name_norm"] = [normalize_name(n) for n in pool_df["business_name"]]
    pool_df["addr_norm"] = [normalize_address(a) for a in pool_df["business_address"]]
    pool_df["country_key"] = [config.normalize_country_key(normalize_country(c)) for c in pool_df["country"]]

    print("Extracting features...", flush=True)
    feat_df = build_training_features(cand_path, s1_sample_df, pool_df, gt_map)

    # Stratified split at entity level to prevent data leakage across pairs
    print("Splitting dataset and training LightGBM...", flush=True)
    unique_entities = s1_sample_df[["entity_id", "country_key"]].drop_duplicates()
    e_ids = unique_entities["entity_id"].astype(str).tolist()
    c_keys = unique_entities["country_key"].astype(str).tolist()
    train_ids, val_ids = train_test_split(
        e_ids,
        test_size=0.20,
        random_state=42,
        stratify=c_keys,
    )
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)

    train_mask = feat_df["source1_entity_id"].isin(train_ids_set)
    val_mask = feat_df["source1_entity_id"].isin(val_ids_set)

    X_train = feat_df.loc[train_mask, FEATURE_COLS]
    y_train = feat_df.loc[train_mask, "label"]
    X_val = feat_df.loc[val_mask, FEATURE_COLS]
    y_val = feat_df.loc[val_mask, "label"]

    val_meta = feat_df.loc[val_mask, ["source1_entity_id", "candidate_id", "label"]].copy()

    print(f"Train set: {len(X_train):,} pairs ({y_train.sum():,} positive)", flush=True)
    print(f"Val set  : {len(X_val):,} pairs ({y_val.sum():,} positive)", flush=True)

    # Class-weighted LightGBM
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    val_meta["pred_prob"] = model.predict_proba(X_val)[:, 1]
    s1_countries_map = dict(zip(s1_sample_df["entity_id"], s1_sample_df["country_key"]))

    # Sweep thresholds to optimize Macro F_0.5 while enforcing 1-to-1 matching
    print("\nOptimizing decision thresholds...", flush=True)
    val_gt_map = {sid: gt_map.get(sid, set()) for sid in val_ids}

    best_thresholds: Dict[str, float] = {}
    distinct_countries = sorted(list(set(unique_entities["country_key"])))

    for c in distinct_countries:
        c_s1_ids = [sid for sid in val_ids if s1_countries_map.get(sid) == c]
        c_gt = {sid: val_gt_map[sid] for sid in c_s1_ids}
        c_val_meta = val_meta[val_meta["source1_entity_id"].isin(set(c_s1_ids))]

        best_th = 0.50
        best_score = -1.0

        for th in np.arange(0.35, 0.85, 0.05):
            th = round(float(th), 2)
            passed = c_val_meta[c_val_meta["pred_prob"] >= th]
            # Greedy 1-to-1 resolution: highest probability pair wins
            passed_sorted = passed.sort_values("pred_prob", ascending=False)
            dedup = passed_sorted.drop_duplicates(subset=["candidate_id"])
            preds = dedup.groupby("source1_entity_id")["candidate_id"].apply(set).to_dict()

            score, _ = evaluate_macro_f05(preds, c_gt, s1_countries_map)
            if score > best_score:
                best_score = score
                best_th = th

        best_thresholds[c] = best_th
        print(f"  Country '{c}': Best Threshold = {best_th:.2f} -> Validation Macro F_0.5 = {best_score:.4f}", flush=True)

    # Evaluate full validation set under best thresholds
    final_val_preds: Dict[str, Set[str]] = {}
    for c in distinct_countries:
        th = best_thresholds[c]
        c_s1_ids = set(sid for sid in val_ids if s1_countries_map.get(sid) == c)
        passed = val_meta[(val_meta["source1_entity_id"].isin(c_s1_ids)) & (val_meta["pred_prob"] >= th)]
        passed_sorted = passed.sort_values("pred_prob", ascending=False).drop_duplicates(subset=["candidate_id"])
        c_preds = passed_sorted.groupby("source1_entity_id")["candidate_id"].apply(set).to_dict()
        final_val_preds.update(c_preds)

    overall_val_f05, country_val_f05 = evaluate_macro_f05(final_val_preds, val_gt_map, s1_countries_map)

    print("\n" + "=" * 60, flush=True)
    print("Validation Performance Summary (Macro F_0.5)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Overall Validation Macro F_0.5: {overall_val_f05:.4f}", flush=True)
    for c, sc in country_val_f05.items():
        print(f"    - {c:<12}: {sc:.4f}", flush=True)
    print("=" * 60, flush=True)

    print("\nFeature Importances:", flush=True)
    importances = model.feature_importances_
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        print(f"  {name:<22}: {imp}", flush=True)

    model_save_path = config.MODELS_DIR / "lgbm_matcher.joblib"
    meta_save_path = config.MODELS_DIR / "model_meta.json"
    joblib.dump(model, model_save_path)
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump({"thresholds": best_thresholds, "features": FEATURE_COLS}, f, indent=2)
    print(f"\nSaved model -> {model_save_path}", flush=True)
    print(f"Saved metadata -> {meta_save_path}", flush=True)


if __name__ == "__main__":
    train_matching_model()
