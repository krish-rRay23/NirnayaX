# NirnayaX End-to-End Triage Sequence

```
[Incident Event] ────────> [PII Redactor / Masker]
                                   │
                                   ▼
                       [Hybrid RAG Knowledge Retrieval]
                       (Dense Vector + Sparse Lexical RRF)
                                   │
                                   ▼
                         [ML Triage Classifier]
                                   │
                                   ▼
                         [Policy & Gating Engine]
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
       [Confidence >= 0.75]                    [Confidence < 0.75]
               │                                       │
               ▼                                       ▼
    [Automated Action / Routing]               [Escalate to Human]
               │                                       │
               └───────────────────┬───────────────────┘
                                   │
                                   ▼
                       [Jira Audit Sync & Metrics]
```
