"""Tests for JSON/JSONL persistence round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest

from nirnayax.data.generator import generate_dataset, generate_incidents
from nirnayax.data.runbooks import build_runbooks
from nirnayax.data.serialization import (
    load_dataset,
    load_incidents_jsonl,
    load_runbooks,
    save_dataset,
    save_incidents_jsonl,
    save_runbooks,
)
from nirnayax.domain.models import DatasetSplit


def test_dataset_json_round_trip(tmp_path: Path) -> None:
    dataset = generate_dataset(60, split=DatasetSplit.TRAIN)
    path = save_dataset(dataset, tmp_path / "ds.json")
    assert path.exists()
    assert load_dataset(path) == dataset


def test_incidents_jsonl_round_trip(tmp_path: Path) -> None:
    incidents = generate_incidents(40, seed=1)
    path = save_incidents_jsonl(incidents, tmp_path / "inc.jsonl")
    loaded = load_incidents_jsonl(path)
    assert loaded == incidents
    assert path.read_text(encoding="utf-8").count("\n") == len(incidents)


def test_runbooks_json_round_trip(tmp_path: Path) -> None:
    runbooks = build_runbooks()
    path = save_runbooks(runbooks, tmp_path / "rb.json")
    assert load_runbooks(path) == runbooks


def test_save_creates_missing_parent_dirs(tmp_path: Path) -> None:
    dataset = generate_dataset(5, split=DatasetSplit.EVAL)
    nested = tmp_path / "a" / "b" / "c" / "ds.json"
    save_dataset(dataset, nested)
    assert nested.exists()


def test_loading_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a dataset"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(bad)
