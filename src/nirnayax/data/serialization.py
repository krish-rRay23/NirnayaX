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

from ..domain.models import DatasetMetadata, DatasetSplit, Incident, IncidentDataset, Runbook
from ..domain.taxonomy import Category, Channel, Priority, Severity

_RUNBOOKS_ADAPTER: TypeAdapter[tuple[Runbook, ...]] = TypeAdapter(tuple[Runbook, ...])


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_dataset(dataset: IncidentDataset, path: str | Path) -> Path:
    """Write an :class:`IncidentDataset` as pretty JSON. Returns the path."""

    path = Path(path)
    _ensure_parent(path)
    path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_tickets_csv(path: str | Path = "data/all_tickets.csv") -> IncidentDataset:
    """Load canonical IT ticket dataset CSV into an IncidentDataset."""

    path = Path(path)
    if not path.exists() and Path("data/all_tickets.csv").exists():
        path = Path("data/all_tickets.csv")

    incidents: list[Incident] = []
    with path.open(encoding="utf-8") as f:
        import csv

        f.seek(0)
        csv_reader = csv.DictReader(f)
        for i, row in enumerate(csv_reader, start=1):
            title = row.get("title") or ""
            if not title or title.lower() == "nan" or len(title.strip()) < 3:
                title = f"IT ticket issue #{i}"
            body = row.get("body") or ""
            if not body or body.lower() == "nan" or len(body.strip()) < 10:
                body = f"Service desk incident record #{i} details."

            raw_cat = str(int(float(row.get("category") or 0)))
            raw_sub1 = str(int(float(row.get("sub_category1") or 0)))
            urgency = int(float(row.get("urgency") or 3))
            impact = int(float(row.get("impact") or 4))

            raw_cat_int = int(raw_cat)
            if raw_cat_int in (1, 3, 4, 6):
                cat = Category.NETWORK
            elif raw_cat_int == 5:
                cat = Category.APPLICATION_DB
            elif raw_cat_int in (7, 10, 11):
                cat = Category.HARDWARE_ACCESS
            else:
                cat = Category.BILLING_OSS

            from ..domain.taxonomy import subcategories_of
            subs = subcategories_of(cat)
            sub = subs[int(raw_sub1) % len(subs)]

            if urgency == 3 and impact >= 3:
                priority = Priority.P1
            elif urgency >= 2 and impact >= 3:
                priority = Priority.P2
            elif urgency >= 1 and impact >= 2:
                priority = Priority.P3
            else:
                priority = Priority.P4

            if urgency == 3:
                sev = Severity.SEV1
            elif urgency == 2:
                sev = Severity.SEV2
            elif urgency == 1:
                sev = Severity.SEV3
            else:
                sev = Severity.SEV4

            channel = (
                Channel.MONITORING
                if int(float(row.get("ticket_type") or 1)) == 1
                else Channel.EMAIL
            )
            svc = f"svc_{row.get('business_service') or '0'}"

            from datetime import UTC, datetime
            ref_time = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)

            incident = Incident(
                incident_id=f"INC-{i:06d}",
                title=title[:160],
                description=body[:2000],
                category=cat,
                subcategory=sub,
                severity=sev,
                priority=priority,
                channel=channel,
                affected_service=svc,
                region="DC-BLR-01",
                reported_at=ref_time,
                tags=(
                    f"raw_cat:{raw_cat}",
                    f"raw_sub1:{raw_sub1}",
                    f"urgency_{urgency}",
                    f"impact_{impact}",
                ),
                customer_impacting=(impact >= 3),
            )
            incidents.append(incident)

    from datetime import UTC, datetime

    ref_time = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
    metadata = DatasetMetadata(
        name=path.name,
        split=DatasetSplit.TRAIN,
        seed=20260901,
        size=len(incidents),
        generated_at=ref_time,
        generator_version="1.0.0-canonical",
    )
    return IncidentDataset(metadata=metadata, incidents=tuple(incidents))


def load_dataset(path: str | Path) -> IncidentDataset:
    """Load and validate an :class:`IncidentDataset` from CSV or JSON."""

    path = Path(path)
    if path.suffix.lower() == ".csv" or not path.exists():
        return load_tickets_csv(path if path.exists() else Path("data/all_tickets.csv"))
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
    "load_tickets_csv",
    "save_dataset",
    "save_incidents_jsonl",
    "save_runbooks",
]
