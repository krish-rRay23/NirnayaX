"""Deterministic synthetic incident generation.

Given a seed, :func:`generate_dataset` produces a byte-for-byte reproducible set
of :class:`~nirnayax.domain.models.Incident` records. Determinism comes from a
single :class:`random.Random` instance advanced in a fixed order; no wall-clock
or global RNG state is consulted. ``reported_at`` timestamps are derived from a
fixed reference clock (:data:`REFERENCE_TIME`) so serialized datasets are stable.

Train and eval splits are produced by seeding with different values
(:data:`DEFAULT_TRAIN_SEED` / :data:`DEFAULT_EVAL_SEED`); they share the same
distribution but never share records.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TypeVar

from ..domain.models import (
    DatasetMetadata,
    DatasetSplit,
    Incident,
    IncidentDataset,
    Signal,
)
from ..domain.taxonomy import (
    Category,
    Channel,
    Priority,
    Severity,
    Subcategory,
    subcategories_of,
)
from . import catalog

GENERATOR_VERSION = "0.1.0"

#: Fixed logical clock; incidents are timestamped within the 90 days before it.
REFERENCE_TIME = datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)
_REPORT_WINDOW_MINUTES = 90 * 24 * 60

DEFAULT_TRAIN_SEED = 20260901
DEFAULT_EVAL_SEED = 77777

#: Train ids start at 1; eval ids start high so the two splits never collide
#: even when merged into a single corpus.
TRAIN_ID_START = 1
EVAL_ID_START = 900_001

_T = TypeVar("_T")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _weighted_choice(rng: random.Random, choices: Sequence[tuple[_T, float]]) -> _T:
    """Pick one item from ``(item, weight)`` pairs using ``rng``."""

    total = sum(weight for _, weight in choices)
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in choices:
        cumulative += weight
        if threshold < cumulative:
            return item
    return choices[-1][0]  # pragma: no cover - float rounding safety net


def _slug(text: str) -> str:
    return _NON_ALNUM.sub("-", text.lower()).strip("-")


def _sample_signals(
    rng: random.Random, specs: Sequence[catalog.SignalSpec]
) -> tuple[tuple[Signal, ...], dict[str, float | int]]:
    """Sample each signal spec, returning the models and a format context."""

    signals: list[Signal] = []
    context: dict[str, float | int] = {}
    for spec in specs:
        if spec.integer:
            value: float | int = rng.randint(int(spec.low), int(spec.high))
        else:
            value = round(rng.uniform(spec.low, spec.high), 1)
        context[spec.name] = value
        signals.append(Signal(name=spec.name, value=float(value), unit=spec.unit))
    return tuple(signals), context


def _derive_priority(severity: Severity, customer_impacting: bool) -> Priority:
    """Map severity to a work-queue priority, escalating for customer impact."""

    base = {
        Severity.SEV1: Priority.P1,
        Severity.SEV2: Priority.P2,
        Severity.SEV3: Priority.P3,
        Severity.SEV4: Priority.P4,
    }[severity]
    if customer_impacting:
        bumped = {Priority.P4: Priority.P3, Priority.P3: Priority.P2, Priority.P2: Priority.P1}
        return bumped.get(base, base)
    return base


def _generate_incident(rng: random.Random, index: int) -> Incident:
    """Generate a single incident. Advances ``rng`` a fixed number of draws."""

    category: Category = _weighted_choice(rng, catalog.CATEGORY_WEIGHTS)
    subcategory: Subcategory = rng.choice(subcategories_of(category))
    template = catalog.SUBCATEGORY_TEMPLATES[subcategory]

    service = rng.choice(template.services)
    region = rng.choice(catalog.REGIONS)
    node = rng.randint(1, 12)
    host = f"{_slug(service)}-{_slug(region)}-n{node:02d}"

    signals, signal_ctx = _sample_signals(rng, template.signals)
    context: dict[str, object] = {"service": service, "region": region, "host": host}
    context.update(signal_ctx)

    title = rng.choice(template.titles).format(**context)
    description = rng.choice(template.descriptions).format(**context)

    severity: Severity = _weighted_choice(rng, template.severity_weights)
    customer_impacting = rng.random() < template.customer_impacting_rate
    priority = _derive_priority(severity, customer_impacting)
    channel: Channel = _weighted_choice(rng, catalog.CATEGORY_CHANNEL_WEIGHTS[category])

    offset = rng.randint(0, _REPORT_WINDOW_MINUTES)
    reported_at = REFERENCE_TIME - timedelta(minutes=offset)

    return Incident(
        incident_id=f"INC-{index:06d}",
        title=title,
        description=description,
        category=category,
        subcategory=subcategory,
        severity=severity,
        priority=priority,
        channel=channel,
        affected_service=service,
        region=region,
        reported_at=reported_at,
        signals=signals,
        tags=template.tags,
        customer_impacting=customer_impacting,
    )


def generate_incidents(
    size: int, seed: int, *, id_start: int = TRAIN_ID_START
) -> tuple[Incident, ...]:
    """Generate ``size`` incidents deterministically from ``seed``.

    Incident ids are assigned sequentially as ``INC-{id_start:06d}`` upward, so
    disjoint ``id_start`` ranges keep splits collision-free when merged. The
    highest id must remain six digits (``INC-999999``).
    """

    if size < 0:
        raise ValueError("size must be non-negative")
    if id_start < 1:
        raise ValueError("id_start must be positive")
    if size and id_start + size - 1 > 999_999:
        raise ValueError("incident ids would exceed the 6-digit INC-###### space")
    rng = random.Random(seed)
    return tuple(_generate_incident(rng, id_start + i) for i in range(size))


def generate_dataset(
    size: int,
    *,
    seed: int | None = None,
    split: DatasetSplit = DatasetSplit.TRAIN,
    name: str | None = None,
) -> IncidentDataset:
    """Generate a fully-formed :class:`IncidentDataset` with provenance metadata.

    If ``seed`` is omitted, the split's default seed is used. ``generated_at`` is
    pinned to :data:`REFERENCE_TIME` so the serialized dataset is reproducible.
    """

    if seed is None:
        seed = DEFAULT_TRAIN_SEED if split is DatasetSplit.TRAIN else DEFAULT_EVAL_SEED
    id_start = TRAIN_ID_START if split is DatasetSplit.TRAIN else EVAL_ID_START
    incidents = generate_incidents(size, seed, id_start=id_start)
    metadata = DatasetMetadata(
        name=name or f"nirnayax-incidents-{split.value}",
        split=split,
        seed=seed,
        size=len(incidents),
        generated_at=REFERENCE_TIME,
        generator_version=GENERATOR_VERSION,
    )
    return IncidentDataset(metadata=metadata, incidents=incidents)


# Re-exported for callers that want to validate template/category coverage.
__all__ = [
    "DEFAULT_EVAL_SEED",
    "DEFAULT_TRAIN_SEED",
    "EVAL_ID_START",
    "GENERATOR_VERSION",
    "REFERENCE_TIME",
    "TRAIN_ID_START",
    "generate_dataset",
    "generate_incidents",
]
