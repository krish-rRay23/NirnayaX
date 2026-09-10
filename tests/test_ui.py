"""Tests for NirnayaX Streamlit UI client and helper functions."""

from __future__ import annotations

from nirnayax.ui.app import NirnayaXAPIClient


def test_ui_api_client_health() -> None:
    client = NirnayaXAPIClient()
    health = client.health_check()
    assert health.get("status") == "ready"
    assert "components" in health


def test_ui_api_client_triage() -> None:
    client = NirnayaXAPIClient()
    res = client.triage(
        title="PostgreSQL Database Connection Pool Exhausted",
        description="Active database connections reached maximum pool limit 100/100",
        service="auth-service",
        tags=["APPLICATION_DB", "CONNECTION_POOL_EXHAUSTION"],
    )
    assert "category" in res
    assert "subcategory" in res
    assert "priority" in res


def test_ui_api_client_diagnose_and_approval_e2e() -> None:
    client = NirnayaXAPIClient()
    res = client.diagnose(
        title="PostgreSQL Database Connection Pool Exhausted",
        description=(
            "Active database connections reached maximum pool limit 100/100; "
            "application queries timing out."
        ),
        service="auth-service",
        tags=["APPLICATION_DB", "CONNECTION_POOL_EXHAUSTION"],
    )
    assert res.get("status") in ("AWAITING_APPROVAL", "RESOLVED", "ESCALATED")
    assert res.get("trace_id") is not None
    assert res.get("incident_id") is not None

    jira_key = res.get("jira_issue_key")
    if res.get("status") == "AWAITING_APPROVAL" and jira_key:
        app_res = client.approve(jira_key, approved_by="oncall-ui-lead")
        assert app_res.get("status") == "RESOLVED"
        assert app_res.get("trace_id") == res.get("trace_id")


def test_ui_api_client_audit_logs() -> None:
    client = NirnayaXAPIClient()
    logs = client.get_audit_logs()
    assert "total_events" in logs
    assert "events" in logs
    assert logs["total_events"] > 0
