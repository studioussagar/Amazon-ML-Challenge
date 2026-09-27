from __future__ import annotations

import gc
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .blocking_tfidf import candidates_for_group

# Lightweight SentenceTransformer model for semantic fallback
MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
SHARD_SIZE: int = 25_000
ENCODE_BATCH_SIZE: int = 512

_model = None


def get_model():
    """Lazy initialization of embedding model."""
    global _model
    if _model is None:
        import torch
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def encode(texts: List[str], batch_size: int = ENCODE_BATCH_SIZE) -> np.ndarray:
    """Encode strings into normalized float32 embeddings."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def build_candidates_embedding(s1: pd.DataFrame, s23: pd.DataFrame,
                               top_k: int = 30,
                               shard_size: int = SHARD_SIZE,
                               batch_size: int = ENCODE_BATCH_SIZE,
                               verbose: bool = False) -> Dict[str, List[str]]:
    """Generate dense embedding candidate matches per block."""
    t0 = time.perf_counter()
    s1 = s1.reset_index(drop=True)
    s23 = s23.reset_index(drop=True)

    out: Dict[str, List[str]] = {}
    base_groups = {k: v.to_numpy() for k, v in s23.groupby("block_key").groups.items()}
    n_groups = 0

    for key, gq_idx in s1.groupby("block_key").groups.items():
        q_idx_arr = gq_idx.to_numpy()
        q_ids: List[str] = s1["entity_id"].iloc[q_idx_arr].tolist()
        gb_idx_arr = base_groups.get(key)
        if gb_idx_arr is None or len(gb_idx_arr) == 0:
            for qid in q_ids:
                out[qid] = []
            continue

        b_ids: List[str] = s23["entity_id"].iloc[gb_idx_arr].tolist()
        q_texts: List[str] = s1["match_text"].iloc[q_idx_arr].tolist()
        b_texts: List[str] = s23["match_text"].iloc[gb_idx_arr].tolist()

        # Encode on demand per block and clean up immediately
        q_vecs = encode(q_texts, batch_size=batch_size)
        b_vecs = encode(b_texts, batch_size=batch_size)

        out.update(candidates_for_group(q_ids, q_vecs, b_ids, b_vecs,
                                        top_k, shard_size))

        del q_vecs, b_vecs, q_texts, b_texts
        n_groups += 1

    if verbose:
        print(f"[embedding] {n_groups} groups, {len(out)} queries, "
              f"{time.perf_counter() - t0:.1f}s")
    return out
