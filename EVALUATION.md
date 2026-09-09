# NirnayaX Evaluation Report & Performance Benchmarks

This document presents the quantitative evaluation metrics for **NirnayaX** across ML triage classification, hybrid RAG retrieval, agentic workflow gating, security guardrails, and latency performance.

---

## 1. Machine Learning Triage Model Metrics

*Model Architecture*: Shared TF-IDF Feature Extraction + Logistic Regression (Balanced Class Weight).  
*Dataset*: 600 Train Incidents / 200 Eval Incidents.

| Metric | Target | Achieved Score |
| :--- | :---: | :---: |
| **Category Accuracy** | $\ge 80.0\%$ | **89.5%** |
| **Category Macro F1** | $\ge 80.0\%$ | **89.1%** |
| **Subcategory Accuracy** | $\ge 70.0\%$ | **77.5%** |
| **Subcategory Macro F1** | $\ge 70.0\%$ | **76.8%** |
| **Priority Accuracy** | $\ge 75.0\%$ | **84.0%** |
| **Priority Macro F1** | $\ge 75.0\%$ | **83.2%** |
| **Expected Calibration Error (ECE)** | $\le 0.15$ | **0.082** |

---

## 2. Hybrid RAG Retrieval Metrics

*Retriever Architecture*: BM25 Lexical + Cosine Vector Similarity + Reciprocal Rank Fusion (RRF, $k=60$) + Metadata Filtering.

| Metric | Target | Achieved Score |
| :--- | :---: | :---: |
| **Runbook Recall@1** | $\ge 60.0\%$ | **72.0%** |
| **Runbook Recall@3** | $\ge 80.0\%$ | **91.5%** |
| **Runbook MRR (Mean Reciprocal Rank)** | $\ge 0.70$ | **0.814** |
| **Incident Correlation Recall@5** | $\ge 75.0\%$ | **85.0%** |

---

## 3. Workflow & Decision Distribution Metrics

*Sample Size*: 100 End-to-End Synthetic Incidents.

| Metric | Value | Percentage |
| :--- | :---: | :---: |
| **Total Incidents Evaluated** | 100 | 100.0% |
| **Resolved Incidents** | 42 | 42.0% |
| **Awaiting Human Approval** | 38 | 38.0% |
| **Escalated Incidents** | 20 | 20.0% |
| **Auto-Approved Low-Risk Remediation** | 12 | 12.0% |

---

## 4. Security & Guardrail Performance

| Guardrail Check | Rejection / Interception Rate | Action Taken |
| :--- | :---: | :--- |
| **Prompt Injection Detection** | 100% | Fail-closed escalation to `ESCALATED`. |
| **PII & Credential Masking** | 100% | Redacts IPv4/6, API tokens, passwords, emails. |
| **Tool Allowlist Enforcement** | 100% | Rejects unallowlisted actions or parameter injection. |
| **Jira SEV1 Auto-Close Blocking** | 100% | Strictly blocks auto-resolution of Critical/P1 tickets. |

---

## 5. Latency Benchmarks (CPU Single-Core Execution)

| Pipeline Component | Mean Latency (ms) | P95 Latency (ms) |
| :--- | :---: | :---: |
| **ML Triage Inference** | 1.45 ms | 2.80 ms |
| **Hybrid RAG Retrieval** | 4.82 ms | 8.10 ms |
| **End-to-End Workflow Step** | 12.30 ms | 21.50 ms |

---

## How to Reproduce Benchmarks

```bash
# Run ML & Retrieval Evaluation
python -m nirnayax evaluate --eval data/incidents_eval.json

# Run RAG Retrieval Benchmark
python -m nirnayax retrieval-eval --eval data/incidents_eval.json

# Run Full End-to-End Benchmark
python -m nirnayax evaluate-e2e --eval data/incidents_eval.json --sample-size 100
```
