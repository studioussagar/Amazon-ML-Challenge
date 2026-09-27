# Business Entity Resolution — Amazon ML Challenge 2026

End-to-end, reproducible solution for the Amazon ML Challenge 2026: Business Entity Resolution across France, the United States, and India.

---

## 1. Directory Structure

```
business_entity_resolution/
├── src/
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Central paths & project configuration
│   ├── text_normalize.py      # Unicode normalization & legal suffix stripping
│   ├── data_loader.py         # Streaming dataset loader with Polars pushdown
│   ├── block_keys.py          # 4-Key inverted index blocking engine
│   ├── blocking.py            # Blocking generation & candidate pooling
│   ├── blocking_tfidf.py      # Sub-word TF-IDF candidate generation
│   ├── blocking_embedding.py  # Semantic embedding candidate generator
│   ├── features.py            # 8-dimensional pairwise feature extractor
│   ├── train_model.py         # Model training & threshold calibration script
│   └── predict_test.py        # End-to-end test streaming inference pipeline
├── models/
│   ├── lgbm_matcher.joblib    # Pre-trained LightGBM classification model
│   └── model_meta.json        # Calibrated decision thresholds (0.80) & feature schema
├── README.md                  # This reproduction guide
└── requirements.txt           # Pinned python dependencies
```

---

## 2. Environment Setup

Python 3.10+ (tested on Python 3.11.9 on Windows / Linux).

```bash
# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install minimal pinned dependencies
pip install -r requirements.txt
```

---

## 3. Reproduction Workflow

### Step A: Model Reproduction (Training from Ground Truth)
A pre-trained model is already included in `models/lgbm_matcher.joblib` and `models/model_meta.json`. To retrain the model from scratch on the training ground truth:

```bash
python -m src.train_model
```
* **Inputs**: `dataset/train/train_source1.tsv`, `dataset/train/train_source2.tsv`, `dataset/train/train_source3.tsv`, `dataset/train/train_ground_truth.tsv`
* **Outputs**: Generates `models/lgbm_matcher.joblib` and `models/model_meta.json`.
* **Validation Score**: Reaches `0.8022 Macro F_0.5` across countries.

---

### Step B: Candidate Blocking & Inference (Producing Submission TSVs)
To generate the final submission files for all 1,732,544 test query entities:

```bash
python -m src.predict_test --chunk-size 2000
```

* **Inputs**: `dataset/test/test_source1.tsv`, `dataset/test/test_source2.tsv`, `dataset/test/test_source3.tsv`, `models/lgbm_matcher.joblib`, `models/model_meta.json`.
* **Execution**:
  1. Partitions by country: France (259k), US (663k), India (810k).
  2. Builds 4-key inverted index blocking in memory (<3.6 GB peak RAM).
  3. Computes 8-feature pairwise matrices in micro-chunks of 2,000 entities.
  4. Applies calibrated conservative threshold (`0.80`) and enforces 1-to-1 graph consistency.
  5. Progressively streams results to output files on disk.
* **Outputs**:
  - `output/matching_results.tsv` (1,732,544 rows + 1 header line, 63.3 MB)
  - `output/candidate_pairs.tsv` (1,732,544 rows + 1 header line, 594.2 MB)

---

### Step C: Verification & Submission Validation
Run the benchmark submission validator:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Expected result:
```
ML Challenge 2026 — submission validator
  test dir: .../dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (584749 empty, 1147795 non-empty).
  candidate_pairs.tsv:  1732544 rows (12 empty, 1732532 non-empty).

PASS — no blocking issues found. Safe to submit.
```
