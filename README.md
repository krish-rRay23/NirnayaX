# NirnayaX — Enterprise L1 IT Incident Triage Engine

[![CI](https://github.com/krish-rRay23/NirnayaX/actions/workflows/ci.yml/badge.svg)](https://github.com/krish-rRay23/NirnayaX/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](#)

NirnayaX is an enterprise-grade L1 IT incident triage and automated diagnosis engine designed for high-reliability telecom, IT service management (ITSM), and cloud operations.

It combines **Machine Learning Triage** (TF-IDF + Logistic Regression trained on canonical data), **Hybrid RAG Retrieval** (BM25 + Vector Search + Reranking), **LangGraph-Style Workflow Execution**, **MCP-Compatible Jira Synchronization**, **Human-in-the-Loop Approval Gates**, a **Policy-Driven Guardrail Security Layer**, and an interactive **Streamlit Demo Dashboard**.

---

## Key Features

- 🎯 **ML Triage Engine**: Predicts incident category, subcategory, and priority with calibrated confidence scores on canonical dataset (`data/all_tickets.csv`).
- 📚 **Hybrid RAG Retrieval**: Combines BM25 lexical search, vector similarity search, reciprocal rank fusion (RRF), and metadata filtering over runbook catalogs and historical incident knowledge.
- ⚙️ **Typed LangGraph Workflow**: State machine graph (`NEW` $\rightarrow$ `TRIAGED` $\rightarrow$ `CORRELATED` $\rightarrow$ `DIAGNOSING` $\rightarrow$ `DECISION` $\rightarrow$ `AWAITING_APPROVAL` $\rightarrow$ `REMEDIATION` $\rightarrow$ `VERIFICATION` $\rightarrow$ `RESOLVED` / `ESCALATED`).
- 🔄 **Atlassian Rovo MCP v2 Jira Integration**: Integrates Jira issues via the official Atlassian Rovo MCP v2 endpoint (`https://mcp.atlassian.com/v2/mcp`) using `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, and `JIRA_PROJECT_KEY`. Explicit offline fallback available via `MockJiraAdapter`. Enforces explicit human approval for `CRITICAL/HIGH` risk actions or `SEV1/P1` incidents. **Critical incidents are never auto-closed**.
- 🛡️ **Security Guardrails & Audit**: Centralized policy engine providing PII sanitization (masking IPs, API keys, credentials, emails), prompt injection detection, tool allowlisting, and fail-closed execution with structured audit logging.
- 💻 **Streamlit Demo Dashboard**: Interactive UI (`src/nirnayax/ui/app.py`) for real-time incident triage, runbook/correlation visualization, visual workflow progression, human approval actions, and structured audit logs.
- 🚀 **Production REST API**: Built with FastAPI, featuring `/healthz`, `/readyz`, `/api/v1/triage`, `/api/v1/retrieve`, `/api/v1/diagnose`, `/api/v1/jira/approve`, `/api/v1/jira/reject`, and `/api/v1/audit/logs`.
- 🐳 **One-Command Docker Setup**: Launch the entire containerized service stack (FastAPI backend + Streamlit frontend) using `docker compose up --build`.

---

## Quickstart

### 1. Local Installation

```bash
# Clone repository
git clone https://github.com/krish-rRay23/NirnayaX.git
cd NirnayaX

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with all optional components (ML, RAG, API, UI, Dev)
pip install -e ".[all]"
```

### 2. Launch Streamlit Interactive Demo UI

```bash
streamlit run src/nirnayax/ui/app.py
```
Visit **http://localhost:8501** in your browser to test golden-path remediation and security fail-closed escalation.

### 3. Run Deterministic CLI Demo

```bash
python -m nirnayax demo
```

### 4. Start FastAPI Production Server

```bash
uvicorn nirnayax.api.app:app --reload --host 0.0.0.0 --port 8000
```
Visit **http://localhost:8000/docs** for the interactive Swagger API documentation.

### 5. Docker One-Command Startup

```bash
docker compose up --build
```
- **Streamlit UI**: http://localhost:8501
- **FastAPI REST API**: http://localhost:8000/docs

---

## CLI Reference

| Command | Description |
| :--- | :--- |
| `nirnayax demo` | Run deterministic end-to-end L1 incident triage, diagnosis, Jira sync, and human approval demo. |
| `nirnayax guardrail-demo` | Demo PII redaction, prompt injection detection, fail-closed escalation, and audit logging. |
| `nirnayax jira-demo` | Demo Jira ticket lifecycle, comment sync, and human-in-the-loop approval. |
| `nirnayax evaluate-e2e` | Run quantitative evaluation across ML, RAG, workflow, and latency benchmarks. |
| `nirnayax train` | Train the ML triage model on canonical `data/all_tickets.csv` and persist model artifacts. |
| `nirnayax predict` | Run ML triage prediction on a single ticket draft. |
| `nirnayax retrieve` | Run hybrid RAG search over runbook knowledge base. |
| `nirnayax serve` | Start production FastAPI REST server. |

---

## System Architecture & Documentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for full technical architecture diagrams, domain models, and state machine transitions.

See [EVALUATION.md](EVALUATION.md) for quantitative benchmark reports and ML baseline metrics.

See [GUARDRAILS.md](GUARDRAILS.md) for security posture and safety enforcement rules.

See [AI_USAGE.md](AI_USAGE.md) for AI capabilities and deterministic control boundaries.

---

## License

Proprietary — NirnayaX Project. All Rights Reserved.
