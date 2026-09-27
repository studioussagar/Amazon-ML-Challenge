from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError as exc:
    raise ImportError("faiss-cpu is required: pip install faiss-cpu") from exc

from sklearn.feature_extraction.text import HashingVectorizer

# Dimension for character n-gram hashing
HASH_DIM: int = 2**12
SHARD_SIZE: int = 25_000
IVF_THRESHOLD: int = 2_048
IVF_NPROBE: int = 16


def make_vectorizer(hash_dim: int = HASH_DIM) -> HashingVectorizer:
    """Create stateless char n-gram vectorizer."""
    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        n_features=hash_dim,
        alternate_sign=False,
        norm="l2",
        dtype=np.float32,
    )


def _to_dense(vec, texts: List[str]) -> np.ndarray:
    mat = vec.transform(texts)
    return np.asarray(mat.toarray(), dtype=np.float32)


def _build_index(base: np.ndarray) -> object:
    """Construct FAISS index (exact FlatIP or IVF for larger shards)."""
    d = base.shape[1]
    n = base.shape[0]
    # For large shards, use IVF to accelerate nearest-neighbor search
    if n >= IVF_THRESHOLD:
        nlist = int(min(64, max(16, int(np.sqrt(n)))))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        train_n = min(n, 20_000)
        rng = np.random.default_rng(0)
        train_idx = rng.choice(n, size=train_n, replace=False)
        index.train(base[train_idx])
        index.add(base)
        index.nprobe = IVF_NPROBE
        return index
    # Exact inner product (cosine on L2-normalized vectors)
    index = faiss.IndexFlatIP(d)
    index.add(base)
    return index


def _search_shard(base_ids: List[str], base_vecs: np.ndarray,
                  query_vecs: np.ndarray, k: int):
    index = _build_index(base_vecs)
    k = min(k, base_vecs.shape[0])
    scores, idx = index.search(query_vecs, k)
    return scores, idx


def candidates_for_group(s1_ids: List[str], query_vecs: np.ndarray,
                         base_ids: List[str], base_vecs: np.ndarray,
                         top_k: int,
                         shard_size: int = SHARD_SIZE) -> Dict[str, List[str]]:
    """Retrieve top-k candidates for queries within one block."""
    n_queries = len(s1_ids)
    best_scores = np.full((n_queries, top_k), -np.inf, dtype=np.float32)
    best_ids: List[List[Optional[str]]] = [[None] * top_k for _ in range(n_queries)]

    if len(base_ids) <= shard_size:
        scores, idx = _search_shard(base_ids, base_vecs, query_vecs, top_k)
        return {
            s1_ids[qi]: [base_ids[int(i)] for i in idx[qi] if i >= 0]
            for qi in range(n_queries)
        }

    # Process large blocks in shards to limit peak RAM
    for start in range(0, len(base_ids), shard_size):
        shard_ids = base_ids[start:start + shard_size]
        shard_vecs = base_vecs[start:start + shard_size]
        scores, idx = _search_shard(shard_ids, shard_vecs, query_vecs, top_k)
        for qi in range(n_queries):
            cand = [(float(scores[qi, j]), shard_ids[int(idx[qi, j])])
                    for j in range(scores.shape[1]) if idx[qi, j] >= 0]
            merged = [(best_scores[qi, j], best_ids[qi][j])
                      for j in range(top_k) if best_ids[qi][j] is not None]
            merged.extend(cand)
            merged.sort(key=lambda t: -t[0])
            merged = merged[:top_k]
            for j, (sc, cid) in enumerate(merged):
                best_scores[qi, j] = sc
                best_ids[qi][j] = cid

    return {
        s1_ids[qi]: [cid for cid in best_ids[qi] if cid is not None]
        for qi in range(n_queries)
    }


def build_candidates_tfidf(s1: pd.DataFrame, s23: pd.DataFrame,
                           top_k: int = 30,
                           hash_dim: int = HASH_DIM,
                           shard_size: int = SHARD_SIZE,
                           verbose: bool = False) -> Dict[str, List[str]]:
    """Generate candidate pairs within shared blocking key partitions."""
    vec = make_vectorizer(hash_dim)
    out: Dict[str, List[str]] = {}
    base_groups = {k: v.to_numpy() for k, v in s23.groupby("block_key").groups.items()}
    t0 = time.perf_counter()
    n_groups = 0
    for key, gq_idx in s1.groupby("block_key").groups.items():
        if not key or key == "__none__" or str(key).endswith("\x1f__none__"):
            continue
        q_idx_arr = gq_idx.to_numpy()
        q_ids: List[str] = s1["entity_id"].iloc[q_idx_arr].tolist()
        q_texts: List[str] = s1["match_text"].iloc[q_idx_arr].tolist()
        gb_idx_arr = base_groups.get(key)
        if gb_idx_arr is None or len(gb_idx_arr) == 0:
            for qid in q_ids:
                if qid not in out:
                    out[qid] = []
            continue
        b_ids: List[str] = s23["entity_id"].iloc[gb_idx_arr].tolist()
        b_texts: List[str] = s23["match_text"].iloc[gb_idx_arr].tolist()
        q_vecs = _to_dense(vec, q_texts)
        b_vecs = _to_dense(vec, b_texts)
        out.update(candidates_for_group(q_ids, q_vecs, b_ids, b_vecs,
                                        top_k, shard_size))
        n_groups += 1
    if verbose:
        print(f"[tfidf] {n_groups} groups, {len(out)} queries, "
              f"{time.perf_counter() - t0:.1f}s")
    return out
