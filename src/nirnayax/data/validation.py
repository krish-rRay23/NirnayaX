"""Dataset- and catalog-level validation.

Per-record invariants (types, id formats, taxonomy consistency, field bounds)
are already enforced by the Pydantic models at construction time. This module
adds *collection-level* checks that a single record cannot express: id
uniqueness, timestamp sanity against the dataset's logical clock, taxonomy
coverage, and — for runbooks — that every subcategory is covered and that
cross-references resolve.

Checks are split into ``errors`` (violations that make the dataset unusable) and
``warnings`` (quality signals worth surfacing but not fatal).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel

from ..domain.models import Incident, IncidentDataset, Runbook
from ..domain.taxonomy import all_subcategories, category_of


class ValidationReport(BaseModel):
    """Outcome of a validation pass."""

    checked: int
    errors: list[str] = []
    warnings: list[str] = []

    @property
    def ok(self) -> bool:
        """True when there are no hard errors (warnings are allowed)."""

        return not self.errors

    def raise_for_status(self) -> None:
        """Raise :class:`ValueError` if the report contains errors."""

        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise ValueError(
                f"dataset validation failed with {len(self.errors)} error(s):\n  - {joined}"
            )

    def render(self) -> str:
        """Human-readable one-block summary."""

        status = "OK" if self.ok else "FAILED"
        lines = [f"Validation: {status} ({self.checked} record(s) checked)"]
        for err in self.errors:
            lines.append(f"  ERROR   {err}")
        for warn in self.warnings:
            lines.append(f"  WARNING {warn}")
        return "\n".join(lines)


def validate_incidents(
    incidents: Sequence[Incident], *, reference_time: datetime | None = None
) -> ValidationReport:
    """Validate a sequence of incidents at the collection level.

    ``reference_time`` is treated as "now"; incidents reported after it are
    flagged as errors. When omitted, the future-check is skipped.
    """

    errors: list[str] = []
    warnings: list[str] = []

    # Unique ids.
    id_counts = Counter(inc.incident_id for inc in incidents)
    for inc_id, count in sorted(id_counts.items()):
        if count > 1:
            errors.append(f"duplicate incident_id {inc_id!r} ({count} occurrences)")

    for inc in incidents:
        # Taxonomy consistency (defence-in-depth; also enforced by the model).
        expected = category_of(inc.subcategory)
        if inc.category != expected:
            errors.append(
                f"{inc.incident_id}: subcategory {inc.subcategory.value} "
                f"maps to {expected.value}, not {inc.category.value}"
            )
        # Timestamp sanity.
        if reference_time is not None and inc.reported_at > reference_time:
            errors.append(
                f"{inc.incident_id}: reported_at {inc.reported_at.isoformat()} is in the future"
            )

    # Coverage warnings.
    seen = {inc.subcategory for inc in incidents}
    missing = [s.value for s in all_subcategories() if s not in seen]
    if missing:
        warnings.append(f"{len(missing)} subcategory(ies) absent from dataset: {sorted(missing)}")

    return ValidationReport(checked=len(incidents), errors=errors, warnings=warnings)


def validate_dataset(dataset: IncidentDataset) -> ValidationReport:
    """Validate an :class:`IncidentDataset`, using its metadata clock as "now"."""

    report = validate_incidents(dataset.incidents, reference_time=dataset.metadata.generated_at)
    if dataset.metadata.size != len(dataset.incidents):
        report.errors.append(
            f"metadata.size ({dataset.metadata.size}) != incident count ({len(dataset.incidents)})"
        )
    return report


def validate_runbooks(runbooks: Sequence[Runbook]) -> ValidationReport:
    """Validate the runbook catalog: unique ids, coverage, resolvable links."""

    errors: list[str] = []
    warnings: list[str] = []

    ids = [rb.runbook_id for rb in runbooks]
    id_set = set(ids)
    id_counts = Counter(ids)
    for rb_id, count in sorted(id_counts.items()):
        if count > 1:
            errors.append(f"duplicate runbook_id {rb_id!r} ({count} occurrences)")

    covered = {rb.subcategory for rb in runbooks}
    for sub in all_subcategories():
        if sub not in covered:
            errors.append(f"no runbook covers subcategory {sub.value}")

    for rb in runbooks:
        for ref in rb.related_runbook_ids:
            if ref not in id_set:
                errors.append(f"{rb.runbook_id}: related_runbook_id {ref!r} does not exist")
            if ref == rb.runbook_id:
                warnings.append(f"{rb.runbook_id}: references itself in related_runbook_ids")

    return ValidationReport(checked=len(runbooks), errors=errors, warnings=warnings)


__all__ = [
    "ValidationReport",
    "validate_dataset",
    "validate_incidents",
    "validate_runbooks",
]
