from __future__ import annotations

import os
from pathlib import Path

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "dataset").exists() and (PROJECT_ROOT.parents[1] / "dataset").exists():
    DATASET_DIR: Path = PROJECT_ROOT.parents[1] / "dataset"
    OUTPUT_DIR: Path = PROJECT_ROOT.parents[1] / "output"
else:
    DATASET_DIR: Path = PROJECT_ROOT / "dataset"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"

TRAIN_DIR: Path = DATASET_DIR / "train"
TEST_DIR: Path = DATASET_DIR / "test"

TRAIN_SOURCE1: Path = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2: Path = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3: Path = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH: Path = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1: Path = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2: Path = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3: Path = TEST_DIR / "test_source3.tsv"

MATCHING_RESULTS_PATH: Path = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH: Path = OUTPUT_DIR / "candidate_pairs.tsv"

INTERIM_DIR: Path = OUTPUT_DIR / "interim"
MODELS_DIR: Path = PROJECT_ROOT / "models" if (PROJECT_ROOT / "models").exists() else PROJECT_ROOT.parents[1] / "models"

SEP: str = "\t"

RECORD_COLUMNS: list[str] = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

GROUND_TRUTH_COLUMNS: list[str] = [
    "source1_entity_id",
    "matched_entity_ids",
]

MATCHING_HEADER: list[str] = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER: list[str] = ["source1_entity_id", "candidate_entity_ids"]

SOURCE_PREFIXES: tuple[str, ...] = ("S1-", "S2-", "S3-")

SOURCE1: str = "source1"
SOURCE2: str = "source2"
SOURCE3: str = "source3"


def normalize_country_key(value: object) -> str:
    """Standardize country label into grouping key."""
    if value is None:
        return "__unknown__"
    text = str(value).strip()
    if not text:
        return "__unknown__"
    return text.lower()


def ensure_output_dirs() -> None:
    for directory in (OUTPUT_DIR, INTERIM_DIR, MODELS_DIR):
        os.makedirs(directory, exist_ok=True)


def enable_utf8_stdout() -> None:
    import sys
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
