"""Tests for dataset and runbook validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nirnayax.data.generator import generate_dataset, generate_incidents
from nirnayax.data.runbooks import build_runbooks
from nirnayax.data.validation import (
    validate_dataset,
    validate_incidents,
    validate_runbooks,
)
from nirnayax.domain.models import DatasetSplit


def test_generated_dataset_validates_clean() -> None:
    report = validate_dataset(generate_dataset(400, split=DatasetSplit.TRAIN))
    assert report.ok
    assert report.errors == []


def test_duplicate_ids_are_flagged() -> None:
    incidents = generate_incidents(5, seed=1)
    report = validate_incidents((*incidents, incidents[0]))
    assert not report.ok
    assert any("duplicate" in err for err in report.errors)


def test_future_timestamp_is_flagged() -> None:
    incidents = generate_incidents(3, seed=1)
    past_reference = datetime(2000, 1, 1, tzinfo=UTC)
    report = validate_incidents(incidents, reference_time=past_reference)
    assert not report.ok
    assert all("future" in err for err in report.errors)


def test_missing_subcategories_warn_not_error() -> None:
    # A tiny sample will not cover every subcategory.
    report = validate_incidents(generate_incidents(3, seed=1))
    assert report.ok  # coverage gaps are warnings only
    assert report.warnings


def test_raise_for_status() -> None:
    incidents = generate_incidents(2, seed=1)
    bad = validate_incidents((*incidents, incidents[0]))
    with pytest.raises(ValueError, match="validation failed"):
        bad.raise_for_status()
    # A clean report does not raise.
    validate_incidents(incidents).raise_for_status()


def test_report_render_mentions_status() -> None:
    report = validate_incidents(generate_incidents(10, seed=1))
    assert "Validation:" in report.render()


def test_runbooks_validate_clean() -> None:
    report = validate_runbooks(build_runbooks())
    assert report.ok


def test_runbook_missing_coverage_is_error() -> None:
    runbooks = build_runbooks()[:-1]  # drop one subcategory's runbook
    report = validate_runbooks(runbooks)
    assert not report.ok
    assert any("no runbook covers" in err for err in report.errors)
