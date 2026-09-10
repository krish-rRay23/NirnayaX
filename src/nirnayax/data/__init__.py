"""Data layer: generation, runbooks, validation, EDA, and persistence."""

from __future__ import annotations

from .eda import EdaReport, LengthStats, compute_eda
from .generator import (
    DEFAULT_EVAL_SEED,
    DEFAULT_TRAIN_SEED,
    EVAL_ID_START,
    GENERATOR_VERSION,
    REFERENCE_TIME,
    TRAIN_ID_START,
    generate_dataset,
    generate_incidents,
)
from .runbooks import build_runbooks
from .serialization import (
    load_dataset,
    load_incidents_jsonl,
    load_runbooks,
    load_tickets_csv,
    save_dataset,
    save_incidents_jsonl,
    save_runbooks,
)
from .validation import (
    ValidationReport,
    validate_dataset,
    validate_incidents,
    validate_runbooks,
)

__all__ = [
    "DEFAULT_EVAL_SEED",
    "DEFAULT_TRAIN_SEED",
    "EVAL_ID_START",
    "GENERATOR_VERSION",
    "REFERENCE_TIME",
    "TRAIN_ID_START",
    "EdaReport",
    "LengthStats",
    "ValidationReport",
    "build_runbooks",
    "compute_eda",
    "generate_dataset",
    "generate_incidents",
    "load_dataset",
    "load_incidents_jsonl",
    "load_runbooks",
    "load_tickets_csv",
    "save_dataset",
    "save_incidents_jsonl",
    "save_runbooks",
    "validate_dataset",
    "validate_incidents",
    "validate_runbooks",
]
