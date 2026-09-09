# NirnayaX AI Usage & Control Boundaries

This document defines the scope of Artificial Intelligence (AI) and Machine Learning (ML) capabilities used in **NirnayaX**, detailing how deterministic control boundaries, gating mechanisms, and human-in-the-loop controls govern AI operations.

---

## 1. Scope of AI & Machine Learning

NirnayaX utilizes targeted, transparent, and reproducible Machine Learning models for initial incident classification and hybrid Information Retrieval (RAG):

1. **ML Incident Triage (`nirnayax.ml`)**:
   - **Architecture**: Shared TF-IDF N-gram feature representation + Logistic Regression multi-target classifiers.
   - **Role**: Predicts incident `category`, `subcategory`, and `priority` from ticket title and description.
   - **Outputs**: Calibrated class predictions and confidence probabilities ($0.0 \dots 1.0$).

2. **Hybrid RAG Retrieval (`nirnayax.retrieval`)**:
   - **Architecture**: Lexical BM25 search + TF-IDF Cosine Vector Search + Reciprocal Rank Fusion (RRF).
   - **Role**: Correlates relevant runbooks and historical incident knowledge for evidence gathering.

---

## 2. Deterministic Control Boundaries & Gating

AI predictions in NirnayaX are **never** allowed to execute autonomous actions without passing deterministic gating checks:

```
[ML Triage Prediction] + [RAG Retrieved Evidence]
                          |
                          v
        +-----------------------------------+
        |     Confidence & Evidence Gate    |
        +-----------------------------------+
                          |
             +------------+------------+
             |                         |
             v                         v
   Score >= Threshold         Score < Threshold
   (Proceed to Risk Gate)     (Immediate Escalation)
```

1. **Confidence Threshold Gating**:
   - Minimum confidence score threshold (default: $0.50$). If ML confidence or RAG evidence match falls below threshold, the decision is forced to `ESCALATE`.
2. **Evidence Match Gate**:
   - Requires at least 1 matching runbook in the catalog for automated remediation.
3. **Guardrail Interception**:
   - Inputs are scanned by `GuardrailEngine` prior to ML inference. Prompt injection or PII violations bypass AI inference and fail closed directly to `ESCALATED`.

---

## 3. Human-in-the-Loop Approval Policy

To maintain safety across critical operations:

- **Critical / SEV1 Incidents**: **Must always require explicit human approval**. The AI engine is strictly prohibited from auto-closing or auto-remediating Critical incidents.
- **High-Risk Actions**: Service restarts, BGP peering resets, connection pool flushes, and infrastructure changes require explicit operator approval via Jira or REST API (`/api/v1/jira/approve`).
- **Low-Risk Actions**: Low-risk diagnostic observations and non-destructive actions may be auto-approved only if explicitly enabled by policy configuration (`auto_approve_low_risk=True`).

---

## 4. Reproducibility & Auditability

- All ML models and data generators use deterministic random seeds (`seed=42`).
- Every workflow step records immutable structured audit logs (`trace_id`, `actor`, `decision`, `confidence`, `evidence`, `guardrail_results`).
