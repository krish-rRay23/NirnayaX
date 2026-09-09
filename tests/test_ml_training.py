"""Unit + integration tests for the training pipeline.

The pipeline must be **reproducible**: the same (dataset, config) yields byte-for-byte
identical predictions and an identical training fingerprint. It must also stamp
honest provenance into the model metadata.
"""

from __future__ import annotations

import pytest

from nirnayax.data import generate_dataset
from nirnayax.domain import DatasetSplit
from nirnayax.domain.models import DatasetMetadata, IncidentDataset
from nirnayax.ml import TARGETS, TicketDraft, TrainingConfig, TriageModel, train_triage_model
from nirnayax.ml.training import MODEL_VERSION


def test_empty_dataset_raises() -> None:
    empty = IncidentDataset(
        metadata=DatasetMetadata(
            name="empty",
            split=DatasetSplit.TRAIN,
            seed=1,
            size=0,
            generated_at=generate_dataset(1, split=DatasetSplit.TRAIN).metadata.generated_at,
            generator_version="0.1.0",
        ),
        incidents=(),
    )
    with pytest.raises(ValueError, match="empty"):
        train_triage_model(empty)


def test_metadata_records_provenance(
    trained_model: TriageModel, train_dataset: IncidentDataset
) -> None:
    meta = trained_model.metadata
    assert meta.model_version == MODEL_VERSION
    assert meta.train_size == len(train_dataset.incidents)
    assert meta.train_dataset_split == "train"
    assert meta.targets == TARGETS
    assert set(meta.label_space) == set(TARGETS)
    assert len(meta.label_space["category"]) == 4
    assert len(meta.label_space["subcategory"]) == 17
    assert len(meta.label_space["priority"]) == 4
    # Provenance strings are populated (not empty).
    assert meta.sklearn_version and meta.numpy_version and meta.python_version
    assert len(meta.train_fingerprint) == 16


def test_training_is_reproducible(train_dataset: IncidentDataset) -> None:
    a = train_triage_model(train_dataset)
    b = train_triage_model(train_dataset)

    # Identical fingerprint and identical config-derived metadata (ignoring the clock).
    assert a.metadata.train_fingerprint == b.metadata.train_fingerprint
    assert a.metadata.model_dump(exclude={"created_at"}) == b.metadata.model_dump(
        exclude={"created_at"}
    )

    # Identical predictions on the same input.
    draft = TicketDraft(title="disk almost full", description="database volume at 97 percent")
    assert a.predict(draft) == b.predict(draft)


def test_fingerprint_differs_across_datasets(train_dataset: IncidentDataset) -> None:
    other = generate_dataset(len(train_dataset.incidents), split=DatasetSplit.EVAL)
    a = train_triage_model(train_dataset)
    b = train_triage_model(other)
    assert a.metadata.train_fingerprint != b.metadata.train_fingerprint


def test_config_overrides_are_persisted(train_dataset: IncidentDataset) -> None:
    config = TrainingConfig(seed=12345, C=1.0)
    model = train_triage_model(train_dataset, config)
    assert model.metadata.training_config.seed == 12345
    assert model.metadata.training_config.C == 1.0


def test_model_version_override(train_dataset: IncidentDataset) -> None:
    model = train_triage_model(train_dataset, model_version="9.9.9")
    assert model.model_version == "9.9.9"
    assert model.metadata.model_version == "9.9.9"
