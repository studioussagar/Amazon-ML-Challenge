# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** YAMI  
**Team Members:** Abhirup Animesh Mandal, Sagar Sujit Samadder, Pranav Pramod Thakur, Shrikant Ashok More  
**Submission Date:** September 2026  

--- 

## 1. Executive Summary

We developed a high-throughput, memory-bounded entity resolution pipeline that accurately matches noisy business entities across three heterogeneous, multilingual data sources (English, Indic scripts, European locales) while strictly adhering to hardware limitations. Our solution couples a 4-key inverted index blocking engine with bounded candidate retrieval and a supervised LightGBM pairwise classifier trained on blocking-generated hard negatives. Calibrating the decision threshold specifically against the competition's primary metric, Macro $F_{0.5}$, the pipeline achieved a validation performance of **0.8022 Macro $F_{0.5}$** across validation splits and achieved **89.57% candidate recall** on a 20,000-entity stratified sample (US: 94.92%, India: 82.15%). The pipeline successfully processed all **1,732,544** test entities across France, the United States, and India in streaming batches under strict memory bounds (**$<3.6$ GB peak RAM**), passing all official submission validator checks with zero blocking issues, zero malformed rows, and zero candidate-subset violations.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across Source 1 (reference catalog), Source 2, and Source 3 revealed several distinct challenges:
- **Cross-Source Heterogeneity & Noise:** Source 1 provides a deduplicated reference source, whereas Sources 2 and 3 exhibit significant OCR noise, non-standardized abbreviations (*e.g.*, "Pvt Ltd", "Corp", "SARL", "SAS", "LLP"), casing differences, and punctuation artifacts.
- **Multilingual Scripts & Transliteration:** The corpus spans multiple scripts (Devanagari, Kannada, Latin). Cross-source records frequently alternate between native Unicode scripts and Latin transliterations.
- **Missing & Reordered Address Fields:** Address fields often omit postal codes, district/state identifiers, or street numbers, or invert premises numbers and localities (*e.g.*, "Door No 183, 41st Cross" vs. "Jayanagar 9th Block").
- **Scale & Strict Computational Boundaries:** With 1,732,544 Source 1 entities and over 10 million combined records in Sources 2 and 3, exhaustive pairwise comparison ($>1.7 \times 10^{13}$ pairs) is mathematically intractable. Standard in-memory Cartesian products trigger fatal Out-Of-Memory (OOM) failures.
- **Asymmetric Metric Penalty ($F_{0.5}$):** The competition evaluation metric weights precision twice as heavily as recall ($\beta = 0.5$). False merges cost twice as much as misses, requiring conservative threshold calibration and strict 1-to-1 graph consistency.
- **Open-Set Generalization:** France appears exclusively in the test set and is completely unseen in training data, necessitating country-agnostic normalization rules.

### 2.2 Solution Strategy
We structured the pipeline into a modular, decoupled architecture:
1. **Deterministic Multilingual Text Normalization:** Unicode NFKC normalization, Latin accent folding to ASCII base forms (preserving Indic combining marks), domain-specific legal suffix expansion, address standardization, and ordinal token isolation.
2. **Multi-Key Inverted Index Blocking:** Inverted indices constructed across 4 orthogonal keys (`name_prefix`, `stripped_prefix`, `addr_key`, and `word2_key`) with bounded retrieval limits to keep candidate pools compact ($\sim 12$ candidates on average).
3. **Vectorized Pairwise Feature Engineering:** Rapid computation of 8 discriminative pairwise features (RapidFuzz normalized Levenshtein and Jaro-Winkler, token Jaccard on names and addresses, character 2–4 $n$-gram `HashingVectorizer` cosine similarity, Soundex phonetic match, country match indicator, and relative length disparity).
4. **Supervised Classification & Metric Calibration:** LightGBM gradient boosted classifier trained on hard negatives with balanced class weighting, coupled with country-calibrated conservative decision thresholds ($\tau = 0.80$).
5. **Greedy 1-to-1 Graph Consistency:** Enforces that each Source 2/3 candidate entity is claimed at most once by the highest-confidence Source 1 query entity, eliminating duplicate entity merges.
6. **Streaming & Resumption Execution:** Country-partitioned execution in micro-chunks of 2,000 entities with automatic checkpoint detection, Windows working set memory trimming, and direct-to-disk streaming serialization.

### 2.3 Architectural Design Choices & Pragmatic Simplifications
1. **Multi-Key Lexical/Syntactic Blocking over Embedding ANN:** Bounded inverted index blocking using 4 complementary keys yielded 89.57% sample recall with near-instant lookup and $<3.6$ GB RAM. Dense embedding index construction across 10M records would have required prohibitive indexing overhead and excessive memory without corresponding recall gains.
2. **Character $n$-gram HashingVectorizer Cosine over Transformer Cross-Encoders:** A 2,048-dimensional character 2–4 $n$-gram hashing vectorizer achieved sub-millisecond pairwise cosine calculations while remaining resilient to typos and OCR noise, avoiding the compute bottleneck of heavy neural models.
3. **Calibrated LightGBM over Complex Stacking:** A single, well-regularized LightGBM classifier with balanced class weights produced robust discrimination. Under the asymmetric $F_{0.5}$ metric, post-processing threshold tuning ($\tau = 0.80$) provided greater precision gains than model ensembling.
4. **Greedy 1-to-1 Consistency over Global Graph Clustering:** Sorting candidate pairs by descending predicted probability and claiming matches greedily guaranteed zero multi-merge conflicts in $O(K \log K)$ time per chunk, satisfying the deduplicated reference requirement of Source 1.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 10\text{M}$ search space while ensuring high recall, we implemented 4 complementary blocking keys with country partitioning:

- **Blocking keys used:**
  1. `k1 (name_prefix)`: First 3 characters of the spaceless normalized business name (top-20 candidates).
  2. `k2 (stripped_prefix)`: First 3 characters of business name after stripping leading noise words and honorifics (*e.g.*, *The, Sri, Shri, M/s, Dr, Center, New*) (top-10 candidates).
  3. `k3 (addr_key)`: Composite key combining extracted house/plot number and the primary distinctive street/locality word, ignoring high-frequency address stopwords (top-10 candidates).
  4. `k4 (word2_key)`: First 3 characters of the second distinctive name token, providing robustness against word reordering.

- **Candidate pairs generated:**
  - Total Source 1 test entities: **1,732,544**
  - Non-empty candidate sets: **1,732,532** (99.999%)
  - Singletons (no candidates retrieved due to empty input fields): **12** (0.001%)
  - Mean candidate pool size: **~12 candidates** per Source 1 entity
  - Output candidate file size: **594.2 MB** (`output/candidate_pairs.tsv`)

- **Ensuring true matches were not lost:**
  By computing the union of orthogonal keys, candidates missed by name variations (*e.g.*, abbreviations, alternate trade names) are captured via address matching (`k3`) or inverted word order (`k4`). On a 20,000-entity stratified sample, this 4-key union achieved **89.57% candidate recall** (US: 94.92%, India: 82.15%).

---

## 4. Matching Model

**Features used (8 discriminative pairwise features):**
- **String Similarity Features:**
  - `name_levenshtein`: RapidFuzz normalized Levenshtein ratio ($[0.0, 1.0]$).
  - `name_jaro_winkler`: RapidFuzz Jaro-Winkler prefix-weighted similarity.
- **Token Overlap Features:**
  - `name_token_jaccard`: Jaccard similarity of normalized business name tokens.
  - `addr_token_jaccard`: Jaccard similarity of normalized business address tokens.
- **Vectorized & Phonetic Features:**
  - `tfidf_cosine`: Cosine similarity of character 2–4 $n$-grams computed via a 2,048-feature `HashingVectorizer`.
  - `phonetic_match`: Binary flag indicating matching 4-character Soundex phonetic codes.
- **Structural Consistency Features:**
  - `country_match`: Binary indicator verifying cross-source country agreement.
  - `length_diff`: Normalized relative string length difference between query and candidate names.

**Model type:** LightGBM Gradient Boosted Decision Tree (`LGBMClassifier`)  
- Hyperparameters: 300 estimators, max depth 6, num leaves 31, learning rate 0.05, subsample 0.8, colsample_bytree 0.8, balanced class weighting.
- Training Data: Derived from training ground truth matches (positives) and non-matching blocking candidates (hard negatives), partitioned via entity-level stratified split (80/20) to eliminate pair leakage.

**Threshold selection method:**
Grid search on held-out validation entities optimizing the competition metric **Macro $F_{0.5}$**. A conservative probability threshold of **0.80** was selected across all partitions. Because a false positive penalizes the score twice as much as a false negative, this threshold filters out borderline candidates and admits only high-precision matches.

---

## 5. Results & Error Analysis

Validation performance on held-out entities:

| Metric | Score | Details |
|---|---|---|
| **Macro $F_{0.5}$ Score** | **0.8022** | Primary competition evaluation metric |
| **India Partition Macro $F_{0.5}$** | **0.8041** | High performance on complex address/transliteration data |
| **US Partition Macro $F_{0.5}$** | **0.8003** | Robust precision across corporate naming variants |
| **Blocking Candidate Recall** | **89.57%** | Evaluated on 20,000-entity stratified sample |
| **Subset Violations** | **0** | 100% of final matches exist in candidate sets |
| **1-to-1 Graph Violations** | **0** | Zero duplicate candidate assignments |

- **Common false positives (wrong merges):**
  1. *Co-located Independent Entities:* Distinct businesses sharing identical commercial buildings, plazas, or industrial estates where names share generic trade tokens (*e.g.*, retail kiosks in the same complex).
  2. *Regional Branch Variations:* Multi-location enterprises with identical brand names operating across nearby street addresses.

- **Common false negatives (missed matches):**
  1. *Severe Phonetic/Transliteration Drift:* Extreme script transliterations where character edit distance exceeded the matching threshold.
  2. *Extreme Acronyms:* Entities known exclusively by abbreviated trade acronyms lacking lexical overlap with the official legal business name.

---

## 6. Conclusion

Our end-to-end entity resolution pipeline demonstrates that multi-key inverted index blocking coupled with vectorized feature extraction and conservative LightGBM thresholding provides state-of-the-art matching precision under strict computational constraints. By enforcing greedy 1-to-1 graph consistency and engineering memory-bounded streaming execution ($<3.6$ GB RAM), we achieved **0.8022 Macro $F_{0.5}$** on validation data, processed all **1,732,544** test entities across France, the United States, and India, and verified full compliance with the official challenge validator with zero blocking errors.

---

## Appendix

### A. Code Artefacts

The complete source code is located in `code/business_entity_resolution/`:

```
code/business_entity_resolution/
├── models/
│   ├── lgbm_matcher.joblib       # Pre-trained LightGBM classification model
│   └── model_meta.json           # Calibrated decision thresholds & feature list
├── src/
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Filepaths and configuration constants
│   ├── text_normalize.py         # Unicode NFKC cleaning & abbreviation expansion
│   ├── data_loader.py            # Streamlined TSV ingestion & memoized caching
│   ├── block_keys.py             # 4-Key inverted index blocking engine
│   ├── blocking.py               # Blocking pipeline and candidate pooling
│   ├── blocking_tfidf.py         # Sub-word TF-IDF candidate generation
│   ├── blocking_embedding.py     # SentenceTransformer dense embedding fallback
│   ├── features.py               # 8-dimensional pairwise feature extractor
│   ├── train_model.py            # Model training & threshold calibration script
│   └── predict_test.py           # End-to-end streaming test prediction pipeline
├── requirements.txt              # Pinned python dependencies
└── README.md                     # Reproduction guide
```

#### Reproducing Results
Run the complete test prediction pipeline:
```bash
python -m code.business_entity_resolution.src.predict_test --chunk-size 2000
```

#### Running Official Submission Validation
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### B. Additional Results

#### 1. Runtime Breakdown by Country Partition (Full Test Pipeline Execution)

| Partition | Entities Processed | Candidates Pool | Partition Runtime |
|---|---|---|---|
| **France** | 259,452 | 871,607 | ~14.2 min |
| **United States** | 663,106 | 4,217,994 | ~38.1 min |
| **India** | 809,986 | 4,879,988 | ~54.8 min |
| **Total Test Pipeline** | **1,732,544 Entities** | **9,969,589 Total Records** | **~107.1 min** |
| **Peak Resident RAM** | **Maximum Memory Allocated** | **$<3.6\text{ GB}$** | **Constant Bound** |

#### 2. Official Validator Execution Log

```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (584749 empty, 1147795 non-empty).
  candidate_pairs.tsv:  1732544 rows (12 empty, 1732532 non-empty).

PASS — no blocking issues found. Safe to submit.
```
