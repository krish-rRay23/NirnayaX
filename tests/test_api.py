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


def test_api_readyz() -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "components" in data


def test_api_triage() -> None:
    payload = {
        "title": "BGP session down on core router",
        "description": "BGP peering flapping on interface edge1",
        "affected_service": "core-router-edge1",
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "subcategory" in data
    assert "priority" in data


def test_api_retrieve() -> None:
    payload = {"query": "BGP flapping peering router", "top_k": 2}
    response = client.post("/api/v1/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "runbooks" in data
    assert "incidents" in data
    assert len(data["runbooks"]) <= 2


def test_api_diagnose() -> None:
    payload = {
        "title": "BGP peering flapping on core router",
        "description": "BGP session down on core-router-edge1",
        "affected_service": "core-router-edge1",
    }
    response = client.post("/api/v1/diagnose", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "history" in data


def test_api_audit_logs() -> None:
    response = client.get("/api/v1/audit/logs")
    assert response.status_code == 200
    data = response.json()
    assert "total_events" in data
    assert "events" in data
