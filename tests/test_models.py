"""Tests for the Pydantic domain models."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from nirnayax.domain.models import (
    DatasetMetadata,
    DatasetSplit,
    Incident,
    IncidentDataset,
    Runbook,
    RunbookStep,
    Signal,
)
from nirnayax.domain.taxonomy import Category, Channel, Priority, Severity, Subcategory

_NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _incident(**overrides: object) -> Incident:
    base: dict[str, object] = {
        "incident_id": "INC-000001",
        "title": "Deadlocks on Billing-DB",
        "description": "Database deadlocks observed on the billing database instance.",
        "category": Category.APPLICATION_DB,
        "subcategory": Subcategory.DEADLOCK,
        "severity": Severity.SEV2,
        "priority": Priority.P2,
        "channel": Channel.MONITORING,
        "affected_service": "Billing-DB",
        "region": "DC-BLR-01",
        "reported_at": _NOW,
    }
    base.update(overrides)
    return Incident(**base)  # type: ignore[arg-type]


def test_valid_incident_builds() -> None:
    inc = _incident(signals=(Signal(name="deadlocks_per_min", value=42.0, unit="count"),))
    assert inc.incident_id == "INC-000001"
    assert inc.signals[0].name == "deadlocks_per_min"


def test_incident_is_frozen() -> None:
    inc = _incident()
    with pytest.raises(ValidationError):
        inc.title = "changed"  # type: ignore[misc]


def test_incident_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _incident(unexpected="nope")


def test_incident_rejects_bad_id_pattern() -> None:
    with pytest.raises(ValidationError):
        _incident(incident_id="INC-1")


def test_incident_taxonomy_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _incident(category=Category.NETWORK, subcategory=Subcategory.DEADLOCK)


def test_runbook_step_order_must_be_contiguous() -> None:
    with pytest.raises(ValidationError):
        Runbook(
            runbook_id="RB-NETWORK-001",
            title="Bad runbook",
            category=Category.NETWORK,
            subcategory=Subcategory.LINK_DOWN,
            summary="A summary long enough to pass validation.",
            symptoms=("symptom",),
            steps=(
                RunbookStep(order=1, action="first"),
                RunbookStep(order=3, action="third"),
            ),
            escalation_team="NOC-Network",
            severity_hint=Severity.SEV1,
            estimated_resolution_minutes=30,
            last_reviewed=date(2026, 7, 15),
        )


def test_dataset_size_must_match_incidents() -> None:
    meta = DatasetMetadata(
        name="x",
        split=DatasetSplit.TRAIN,
        seed=1,
        size=5,  # deliberately wrong
        generated_at=_NOW,
        generator_version="0.1.0",
    )
    with pytest.raises(ValidationError):
        IncidentDataset(metadata=meta, incidents=(_incident(),))
