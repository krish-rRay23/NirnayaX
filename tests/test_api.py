"""Tests for FastAPI REST API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from nirnayax.api import app

client = TestClient(app)


def test_api_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "nirnayax"
    assert data["version"] == "0.1.0"


def test_api_readyz() -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "components" in data
    assert data["components"]["ml_triage"] == "ready"
    assert data["components"]["runbook_retriever"] == "ready"
    assert data["components"]["incident_retriever"] == "ready"
    assert data["components"]["diagnosis_workflow"] == "ready"
    assert data["components"]["guardrails"] == "ready"


def test_api_triage() -> None:
    payload = {
        "title": "Database connection pool exhausted on auth service",
        "description": "High connection pool utilization causing 500 error spikes on auth service",
        "affected_service": "auth-service",
        "tags": ["APPLICATION_DB", "CONNECTION_POOL_EXHAUSTION"],
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "subcategory" in data
    assert "priority" in data


def test_api_retrieve() -> None:
    payload = {"query": "database connection pool exhaustion postgres", "top_k": 2}
    response = client.post("/api/v1/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "runbooks" in data
    assert "incidents" in data
    assert len(data["runbooks"]) <= 2


def test_api_app_db_incident_and_human_approval_e2e() -> None:
    payload = {
        "title": "PostgreSQL Database Connection Pool Exhausted",
        "description": (
            "Active database connections reached maximum pool limit 100/100; "
            "application queries timing out."
        ),
        "affected_service": "auth-service",
        "tags": ["APPLICATION_DB", "CONNECTION_POOL_EXHAUSTION"],
    }
    response = client.post("/api/v1/diagnose", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("AWAITING_APPROVAL", "RESOLVED")
    assert data["trace_id"] is not None
    assert data["incident_id"] is not None
    assert data["jira_issue_key"] is not None

    trace_id = data["trace_id"]
    issue_key = data["jira_issue_key"]

    if data["status"] == "AWAITING_APPROVAL":
        approve_payload = {
            "jira_issue_key": issue_key,
            "approved_by": "oncall-lead",
        }
        approve_resp = client.post("/api/v1/jira/approve", json=approve_payload)
        assert approve_resp.status_code == 200
        approve_data = approve_resp.json()
        assert approve_data["status"] == "RESOLVED"
        assert approve_data["trace_id"] == trace_id
        assert approve_data["incident_id"] == issue_key


def test_api_app_db_incident_and_rejection_e2e() -> None:
    payload = {
        "title": "Database connection pool exhausted on auth service",
        "description": "PostgreSQL database connection pool exhausted causing 500 errors",
        "affected_service": "auth-service",
        "tags": ["APPLICATION_DB", "CONNECTION_POOL_EXHAUSTION"],
    }
    response = client.post("/api/v1/diagnose", json=payload)
    assert response.status_code == 200
    data = response.json()

    issue_key = data["jira_issue_key"]
    if data["status"] == "AWAITING_APPROVAL":
        reject_payload = {
            "jira_issue_key": issue_key,
            "approved_by": "oncall-lead",
        }
        reject_resp = client.post("/api/v1/jira/reject", json=reject_payload)
        assert reject_resp.status_code == 200
        reject_data = reject_resp.json()
        assert reject_data["status"] == "ESCALATED"


def test_api_audit_logs() -> None:
    response = client.get("/api/v1/audit/logs")
    assert response.status_code == 200
    data = response.json()
    assert "total_events" in data
    assert "events" in data
    assert data["total_events"] > 0
    first_event = data["events"][0]
    assert "trace_id" in first_event
    assert "incident_id" in first_event
    assert "actor" in first_event
    assert "decision" in first_event
    assert "confidence" in first_event
    assert "evidence" in first_event
    assert "tool_action" in first_event
    assert "status" in first_event
    assert "timestamp" in first_event

