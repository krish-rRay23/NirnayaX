"""Typed domain models for NirnayaX.

All models are **immutable** (``frozen=True``) and **strict** (``extra="forbid"``)
so that a constructed object is a validated, hashable value: taxonomy consistency,
id formats, and field bounds are enforced at construction time rather than trusted
by convention. This keeps the data foundation trustworthy for every layer built
on top of it.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .taxonomy import (
    Category,
    Channel,
    IncidentStatus,
    Priority,
    Severity,
    Subcategory,
    category_of,
)

INCIDENT_ID_PATTERN = r"^INC-\d{6}$"
RUNBOOK_ID_PATTERN = r"^RB-[A-Z_]+-\d{3}$"


class _Frozen(BaseModel):
    """Base config: immutable, strict, and enum values on dump."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)


class Signal(_Frozen):
    """A single observability signal attached to an incident.

    Examples: ``Signal(name="p99_latency_ms", value=1840.0, unit="ms")`` or
    ``Signal(name="disk_used_pct", value=96.4, unit="%")``.
    """

    name: str = Field(min_length=1, max_length=64)
    value: float
    unit: str = Field(default="", max_length=16, description="e.g. 'ms', '%', 'count'")


class Incident(_Frozen):
    """A single L1 incident record.

    ``category``/``subcategory`` are the ground-truth triage labels; a model
    validator guarantees the subcategory actually belongs to the category.
    """

    incident_id: str = Field(pattern=INCIDENT_ID_PATTERN)
    title: str = Field(min_length=3, max_length=160)
    description: str = Field(min_length=10, max_length=2000)
    category: Category
    subcategory: Subcategory
    severity: Severity
    priority: Priority
    status: IncidentStatus = IncidentStatus.NEW
    channel: Channel
    affected_service: str = Field(min_length=1, max_length=80)
    region: str = Field(min_length=1, max_length=40)
    reported_at: datetime
    signals: tuple[Signal, ...] = ()
    tags: tuple[str, ...] = ()
    customer_impacting: bool = False

    @model_validator(mode="after")
    def _check_taxonomy(self) -> Incident:
        expected = category_of(self.subcategory)
        if self.category != expected:
            raise ValueError(
                f"subcategory {self.subcategory.value!r} belongs to "
                f"{expected.value!r}, not {self.category.value!r}"
            )
        return self


class RunbookStep(_Frozen):
    """One ordered troubleshooting action within a runbook."""

    order: int = Field(ge=1)
    action: str = Field(min_length=1, max_length=400)
    expected_signal: str | None = Field(default=None, max_length=200)


class Runbook(_Frozen):
    """A synthetic troubleshooting runbook for a subcategory, with metadata.

    Runbooks form the (future) retrieval knowledge base. Each targets exactly one
    subcategory and carries operational metadata (owning team, severity hint,
    expected resolution time, review date, version).
    """

    runbook_id: str = Field(pattern=RUNBOOK_ID_PATTERN)
    title: str = Field(min_length=3, max_length=160)
    category: Category
    subcategory: Subcategory
    summary: str = Field(min_length=10, max_length=600)
    symptoms: tuple[str, ...] = Field(min_length=1)
    steps: tuple[RunbookStep, ...] = Field(min_length=1)
    escalation_team: str = Field(min_length=1, max_length=60)
    severity_hint: Severity
    estimated_resolution_minutes: int = Field(ge=1, le=10_080)
    tags: tuple[str, ...] = ()
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    last_reviewed: date
    related_runbook_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_consistency(self) -> Runbook:
        expected = category_of(self.subcategory)
        if self.category != expected:
            raise ValueError(
                f"subcategory {self.subcategory.value!r} belongs to "
                f"{expected.value!r}, not {self.category.value!r}"
            )
        orders = [step.order for step in self.steps]
        if orders != list(range(1, len(orders) + 1)):
            raise ValueError(f"runbook steps must be ordered 1..n, got {orders}")
        return self


class DatasetSplit(StrEnum):
    """Which split a dataset represents."""

    TRAIN = "train"
    EVAL = "eval"


class DatasetMetadata(_Frozen):
    """Provenance for a generated dataset (enables reproducibility auditing)."""

    name: str = Field(min_length=1, max_length=80)
    split: DatasetSplit
    seed: int
    size: int = Field(ge=0)
    generated_at: datetime
    generator_version: str
    schema_version: str = "1"


class IncidentDataset(_Frozen):
    """A generated dataset: provenance metadata plus the incident records."""

    metadata: DatasetMetadata
    incidents: tuple[Incident, ...]

    @model_validator(mode="after")
    def _check_size(self) -> IncidentDataset:
        if self.metadata.size != len(self.incidents):
            raise ValueError(
                f"metadata.size ({self.metadata.size}) != number of incidents "
                f"({len(self.incidents)})"
            )
        return self
