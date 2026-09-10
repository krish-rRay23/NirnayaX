# NirnayaX Architecture Specification

This document provides a detailed technical description of the **NirnayaX** enterprise L1 IT incident triage, diagnosis, and resolution architecture.

---

## High-Level System Architecture

```
                  +-----------------------------------+
                  |   FastAPI REST API / CLI Layer    |
                  +-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                           Guardrail Engine                            |
|  - Input Validation & Prompt Injection Detection                      |
|  - PII Protection & Credential Masking (IPv4/6, API Keys, Secrets)    |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                    Diagnosis Workflow State Machine                   |
|                                                                       |
|  [NEW] -> [TRIAGED] -> [CORRELATED] -> [DIAGNOSING] -> [DECISION]     |
|                                                            |          |
|  [RESOLVED] <- [VERIFICATION] <- [REMEDIATION] <- [AWAITING_APPROVAL] |
|       ^                                                    |          |
|       +--------------------- [ESCALATED] <-----------------+          |
+-----------------------------------------------------------------------+
            |                       |                       |
            v                       v                       v
+----------------------+ +--------------------+ +-----------------------+
|   ML Triage Engine   | | Hybrid RAG Engine  | | MCP Jira Sync Adapter |
|  - Category          | | - BM25 Lexical     | | - Issue Creation      |
|  - Subcategory       | | - Vector Cosine    | | - RAG Evidence Sync |
|  - Priority          | | - RRF Fusion       | | - Status Transitions  |
+----------------------+ +--------------------+ +-----------------------+
```

---

## Workflow State Machine Transitions

| State | Role / Operation | Next States |
| :--- | :--- | :--- |
| `NEW` | Ingests ticket draft, sanitizes PII, verifies prompt injection guardrails. | `TRIAGED`, `ESCALATED` |
| `TRIAGED` | Invokes ML Triage Model to predict category, subcategory, and priority with confidence scores. | `CORRELATED` |
| `CORRELATED` | Queries Hybrid RAG Retriever for top-k runbooks and similar historical incidents. | `DIAGNOSING` |
| `DIAGNOSING` | Executes simulated diagnostic tools (`check_service_metrics`, `fetch_service_logs`). | `DECISION` |
| `DECISION` | Evaluates Confidence Gate (combines ML confidence, runbook alignment, diagnostic match). | `AWAITING_APPROVAL`, `REMEDIATION`, `ESCALATION` |
| `AWAITING_APPROVAL` | Creates human approval request. Waits for explicit operator approval if risk is HIGH/CRITICAL or SEV1. | `REMEDIATION`, `ESCALATED` |
| `REMEDIATION` | Validates tool allowlisting policy and executes simulated remediation action (`execute_remediation`). | `VERIFICATION`, `ESCALATED` |
| `VERIFICATION` | Evaluates recovery observation (`verify_service_recovery`). Checks Jira auto-close safety rules. | `RESOLVED`, `ESCALATED` |
| `RESOLVED` | Terminal state: Incident resolved and Jira status transitioned to Resolved. | None |
| `ESCALATED` | Terminal state: Incident escalated to L2 engineering team and Jira status set to Escalated. | None |

---

## Core Domain Principles

1. **Deterministic Execution**:
   - Random seeds are fixed in data generation, ML training, and RAG indexing.
   - Diagnostic observations and remediation tools follow observe-act-verify semantics without non-deterministic side effects.

2. **Clean Boundary Provider Pattern**:
   - The Jira integration uses `JiraAdapter` protocol. Implementations include `MockJiraAdapter` (in-memory, thread-safe for offline testing) and `MCPJiraAdapter` (connecting directly to the official Atlassian Rovo MCP v2 JSON-RPC endpoint at `https://mcp.atlassian.com/v2/mcp`).

3. **Fail-Closed Safety Design**:
   - Any prompt injection attempt, unallowlisted tool invocation, low-confidence prediction, or prohibited Jira auto-close immediately triggers a fail-closed transition to `ESCALATED`.
