"""JSON/JSONL persistence for datasets and runbooks.

Thin, dependency-free helpers built on Pydantic's own JSON (de)serialization so
round-trips are lossless and validated on load. Datasets are stored as a single
pretty-printed JSON object (metadata + incidents); incidents can additionally be
streamed as JSONL, and the runbook catalog is stored as a JSON array.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import TypeAdapter

from ..domain.models import Incident, IncidentDataset, Runbook

_RUNBOOKS_ADAPTER: TypeAdapter[tuple[Runbook, ...]] = TypeAdapter(tuple[Runbook, ...])


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_dataset(dataset: IncidentDataset, path: str | Path) -> Path:
    """Write an :class:`IncidentDataset` as pretty JSON. Returns the path."""

    path = Path(path)
    _ensure_parent(path)
    path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_dataset(path: str | Path) -> IncidentDataset:
    """Load and validate an :class:`IncidentDataset` from JSON."""

    text = Path(path).read_text(encoding="utf-8")
    return IncidentDataset.model_validate_json(text)


def save_incidents_jsonl(incidents: Iterable[Incident], path: str | Path) -> Path:
    """Write incidents as JSON Lines (one incident per line). Returns the path."""

    path = Path(path)
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for incident in incidents:
            handle.write(incident.model_dump_json())
            handle.write("\n")
    return path


def load_incidents_jsonl(path: str | Path) -> tuple[Incident, ...]:
    """Load incidents from a JSON Lines file."""

    incidents: list[Incident] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                incidents.append(Incident.model_validate_json(line))
    return tuple(incidents)


def save_runbooks(runbooks: Sequence[Runbook], path: str | Path) -> Path:
    """Write the runbook catalog as a pretty JSON array. Returns the path."""

    path = Path(path)
    _ensure_parent(path)
    payload = _RUNBOOKS_ADAPTER.dump_python(tuple(runbooks), mode="json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_runbooks(path: str | Path) -> tuple[Runbook, ...]:
    """Load and validate the runbook catalog from a JSON array."""

    text = Path(path).read_text(encoding="utf-8")
    return _RUNBOOKS_ADAPTER.validate_json(text)


__all__ = [
    "load_dataset",
    "load_incidents_jsonl",
    "load_runbooks",
    "save_dataset",
    "save_incidents_jsonl",
    "save_runbooks",
]
