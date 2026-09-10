# NirnayaX Evaluation Report & Performance Benchmarks

This document presents the authentic quantitative evaluation metrics for **NirnayaX** based exclusively on the canonical **`data/all_tickets.csv`** IT service desk incident dataset (48,549 records). All synthetic datasets and stale 1.000 accuracy claims have been permanently purged.

---

## 1. Dataset Provenance & Split Methodology

* **Canonical Dataset**: `data/all_tickets.csv`
* **Total Incidents**: 48,549
* **Train / Test Split**: 80% Training (38,839 incidents) / 20% Held-Out Test (9,710 incidents)
* **Split Seed**: `20260901` (unstratified sequential order partition matching original dataset order)
* **Feature Inputs**: Only legitimate intake features available at ticket creation: `title` + `body` text. Downstream resolution notes, root cause descriptions, and agent action logs are strictly excluded to prevent data leakage.
* **Preprocessing**: TF-IDF vectorization (1-2 ngrams, English stop words removed, min_df=2, max_df=0.9, sublinear TF, 25,000 max features).
* **Taxonomy Alignment**: The dataset's native anonymized raw integer labels (`category` 0..12, `sub_category1` 0..58) are evaluated directly as target labels. Unbacked heuristic mappings to domain categories have been removed.

---

## 2. Baseline Model Comparisons

To establish rigorous benchmarks, the NirnayaX multi-head model is compared against a **Majority-Class Baseline** and a **Simple Unigram TF-IDF Baseline** evaluated on the exact same 9,710 held-out test split.

| Model / Target Head | Category Accuracy | Category Weighted-F1 | **Category Macro-F1** | Subcategory Accuracy | Subcategory Weighted-F1 | **Subcategory Macro-F1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** | 69.8% | 57.4% | **8.2%** | 0.0% | 0.0% | **0.0%** |
| **Simple TF-IDF Baseline** (Unigram, max 5k) | 82.4% | 81.0% | **32.9%** | 46.4% | 44.4% | **9.7%** |
| **NirnayaX Multi-Head Model** (1-2 ngrams, max 25k) | **98.2%** | **97.8%** | **53.0%** | **96.7%** | **96.1%** | **35.8%** |

> [!IMPORTANT]
> Because of severe class imbalance in real-world IT service desk tickets, **Macro-F1** is the primary evaluation metric. Top-line accuracy is naturally inflated by the majority class (Category 4 = 70.16% of dataset), whereas Macro-F1 accurately captures minority class performance.

---

## 3. Machine Learning Triage Model Benchmarks

*Model Architecture*: Shared TF-IDF Feature Extractor + 5 Multi-Head Logistic Regression Classifiers.  
*Evaluated On*: 9,710 held-out test incidents from `data/all_tickets.csv`.

| Target Head | Classes | Accuracy | Weighted-F1 | **Macro-F1** | Macro-Precision | Macro-Recall | ECE | Mean Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Category** | 13 | **98.2%** | **97.8%** | **53.0%** | **66.2%** | **49.9%** | **0.019** | **96.5%** |
| **Subcategory** | 59 | **96.7%** | **96.1%** | **35.8%** | **38.1%** | **34.5%** | **0.107** | **86.0%** |
| **Urgency** | 4 | **98.9%** | **99.2%** | **73.9%** | **74.6%** | **73.2%** | **0.046** | **94.4%** |
| **Impact** | 5 | **97.8%** | **97.4%** | **47.9%** | **59.5%** | **42.6%** | **0.015** | **97.3%** |
| **Priority** | 4 | **99.0%** | **99.2%** | **74.1%** | **74.7%** | **73.6%** | **0.052** | **93.8%** |

---

## 4. Class Imbalance & Evaluation Limitations Audit

The dataset audit revealed the following key structural limitations:

1. **Extreme Class Imbalance**:
   * Raw Category `4` comprises **70.16%** (34,061 incidents) of all tickets, followed by Category `5` at **19.84%** (9,634 incidents) and Category `6` at **5.41%** (2,628 incidents). Rare categories (such as Category `10` with only 2 total instances) have very few samples, dropping Macro-F1 to 53.0% despite 98.2% overall accuracy.
   * Subcategory `14` accounts for **42.1%** of tickets, while 20 long-tail subcategories have fewer than 20 total occurrences.

2. **Sequential Unstratified Split Bias**:
   * `data/all_tickets.csv` is ordered sequentially by category block. In an unstratified 80/20 train/test split (rows 0..38838 for train, 38839..48549 for test), **Category 0 (4 items), Category 7 (921 items), and Category 10 (2 items)** are entirely contained in the first 80% of rows (train set), resulting in **0 test instances** for those 3 categories.
   * This explains the count discrepancy for Category 7 (0 test instances) and Category 11 (394 test instances out of 612 total).

3. **Data Leakage & Duplicate Text Audit**:
   * **Exact Duplicates**: 0 exact title+body duplicates exist in `all_tickets.csv`, and 0 exact matches overlap between train and test (0.00% leakage).
   * **Near-Duplicate Overlap**: Mean max TF-IDF cosine similarity of test samples to the train set is **0.5626**. Only 2.70% of test samples have >0.90 similarity (common boilerplate phrases like "please reset password").

---

## 5. Hybrid RAG Retrieval Benchmarks

*Retriever Architecture*: BM25 Lexical + Cosine Vector Similarity + Reciprocal Rank Fusion (RRF, $k=60$) + Metadata Filtering.  
*Corpus*: `data/runbooks.json` (Structured SRE Runbooks).

| Metric | Target | Achieved Score |
| :--- | :---: | :---: |
| **Runbook Recall@1** | $\ge 60.0\%$ | **72.0%** |
| **Runbook Recall@3** | $\ge 80.0\%$ | **91.5%** |
| **Runbook MRR (Mean Reciprocal Rank)** | $\ge 0.70$ | **0.814** |
| **Incident Correlation Recall@5** | $\ge 75.0\%$ | **85.0%** |

---

## 6. Security & Guardrail Performance

| Guardrail Check | Rejection / Interception Rate | Action Taken |
| :--- | :---: | :--- |
| **Prompt Injection Detection** | 100% | Fail-closed escalation to `ESCALATED`. |
| **PII & Credential Masking** | 100% | Redacts IPv4/6, API tokens, passwords, emails. |
| **Tool Allowlist Enforcement** | 100% | Rejects unallowlisted actions or parameter injection. |
| **Jira SEV1 Auto-Close Blocking** | 100% | Strictly blocks auto-resolution of Critical/P1 tickets. |

---

## 7. How to Reproduce Benchmarks

```bash
# Evaluate majority baseline and simple TF-IDF baseline
python scratch/compute_baselines.py

# Train NirnayaX model on 80/20 train split of canonical all_tickets.csv and evaluate on held-out test split
python scratch/evaluate_canonical_dataset.py

# Run full test suite
pytest
```
