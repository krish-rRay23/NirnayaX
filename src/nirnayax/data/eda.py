"""Lightweight exploratory data analysis over incident datasets.

Deliberately dependency-free (``collections`` + ``statistics`` only) so the core
package stays minimal. :func:`compute_eda` returns a typed :class:`EdaReport`
that is both machine-readable (for tests/dashboards) and renderable as a compact
text summary for the CLI.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel

from ..domain.models import Incident
from ..domain.taxonomy import category_of


class LengthStats(BaseModel):
    """Summary statistics for a text field's character length."""

    min: int
    max: int
    mean: float
    median: float


class EdaReport(BaseModel):
    """Structured EDA summary for a set of incidents."""

    total: int
    by_category: dict[str, int]
    by_subcategory: dict[str, int]
    by_severity: dict[str, int]
    by_priority: dict[str, int]
    by_channel: dict[str, int]
    by_region: dict[str, int]
    customer_impacting: int
    customer_impacting_pct: float
    title_length: LengthStats | None
    description_length: LengthStats | None
    earliest_reported_at: str | None
    latest_reported_at: str | None

    def render(self) -> str:
        """Render a compact, human-readable multi-line summary."""

        if self.total == 0:
            return "EDA: empty dataset (0 incidents)"

        lines = [f"EDA summary - {self.total} incident(s)"]

        def _section(title: str, counts: dict[str, int]) -> None:
            lines.append(f"  {title}:")
            for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                pct = 100.0 * count / self.total
                lines.append(f"    {key:<28} {count:>6}  ({pct:5.1f}%)")

        _section("by category", self.by_category)
        _section("by subcategory", self.by_subcategory)
        _section("by severity", self.by_severity)
        _section("by priority", self.by_priority)
        _section("by channel", self.by_channel)
        lines.append(
            f"  customer-impacting: {self.customer_impacting} ({self.customer_impacting_pct:.1f}%)"
        )
        if self.description_length is not None:
            dl = self.description_length
            lines.append(
                f"  description length: min={dl.min} max={dl.max} "
                f"mean={dl.mean:.1f} median={dl.median:.1f}"
            )
        if self.earliest_reported_at and self.latest_reported_at:
            lines.append(
                f"  reported_at span: {self.earliest_reported_at} to {self.latest_reported_at}"
            )
        return "\n".join(lines)


def _length_stats(values: Sequence[int]) -> LengthStats | None:
    if not values:
        return None
    return LengthStats(
        min=min(values),
        max=max(values),
        mean=round(statistics.fmean(values), 2),
        median=statistics.median(values),
    )


def compute_eda(incidents: Sequence[Incident]) -> EdaReport:
    """Compute an :class:`EdaReport` over ``incidents``."""

    total = len(incidents)
    if total == 0:
        return EdaReport(
            total=0,
            by_category={},
            by_subcategory={},
            by_severity={},
            by_priority={},
            by_channel={},
            by_region={},
            customer_impacting=0,
            customer_impacting_pct=0.0,
            title_length=None,
            description_length=None,
            earliest_reported_at=None,
            latest_reported_at=None,
        )

    by_category: Counter[str] = Counter()
    by_subcategory: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    by_priority: Counter[str] = Counter()
    by_channel: Counter[str] = Counter()
    by_region: Counter[str] = Counter()
    title_lengths: list[int] = []
    description_lengths: list[int] = []
    customer_impacting = 0

    for inc in incidents:
        # Trust the label but normalise category from the taxonomy for safety.
        by_category[category_of(inc.subcategory).value] += 1
        by_subcategory[inc.subcategory.value] += 1
        by_severity[inc.severity.value] += 1
        by_priority[inc.priority.value] += 1
        by_channel[inc.channel.value] += 1
        by_region[inc.region] += 1
        title_lengths.append(len(inc.title))
        description_lengths.append(len(inc.description))
        if inc.customer_impacting:
            customer_impacting += 1

    timestamps = [inc.reported_at for inc in incidents]

    return EdaReport(
        total=total,
        by_category=dict(by_category),
        by_subcategory=dict(by_subcategory),
        by_severity=dict(by_severity),
        by_priority=dict(by_priority),
        by_channel=dict(by_channel),
        by_region=dict(by_region),
        customer_impacting=customer_impacting,
        customer_impacting_pct=round(100.0 * customer_impacting / total, 2),
        title_length=_length_stats(title_lengths),
        description_length=_length_stats(description_lengths),
        earliest_reported_at=min(timestamps).isoformat(),
        latest_reported_at=max(timestamps).isoformat(),
    )


__all__ = ["EdaReport", "LengthStats", "compute_eda"]
