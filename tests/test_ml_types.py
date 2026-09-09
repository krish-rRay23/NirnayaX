"""Unit tests for the ML data contracts (predictions, metadata, metrics).

These models are deliberately free of scikit-learn, so they can be constructed
and asserted on directly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nirnayax.ml import (
    ClassPrediction,
    FeatureConfig,
    ModelMetadata,
    TrainingConfig,
    TriagePrediction,
)


def _class_pred(label: str = "NETWORK", confidence: float = 0.7) -> ClassPrediction:
    return ClassPrediction(
        label=label,
        confidence=confidence,
        distribution=((label, confidence), ("OTHER", 1.0 - confidence)),
    )


def test_class_prediction_top_k_and_render() -> None:
    pred = ClassPrediction(
        label="P2",
        confidence=0.5,
        distribution=(("P2", 0.5), ("P3", 0.3), ("P1", 0.15), ("P4", 0.05)),
    )
    assert pred.top_k(2) == (("P2", 0.5), ("P3", 0.3))
    assert pred.render() == "P2 (50.0%)"


def test_confidence_must_be_a_probability() -> None:
    with pytest.raises(ValidationError):
        _class_pred(confidence=1.5)
    with pytest.raises(ValidationError):
        _class_pred(confidence=-0.1)


def test_class_prediction_is_frozen() -> None:
    pred = _class_pred()
    with pytest.raises(ValidationError):
        pred.label = "OTHER"  # type: ignore[misc]


def test_triage_prediction_render_flags_inconsistency() -> None:
    good = TriagePrediction(
        category=_class_pred("NETWORK"),
        subcategory=_class_pred("BGP_ROUTING"),
        priority=_class_pred("P2"),
        taxonomy_consistent=True,
        model_version="0.1.0",
    )
    assert "mismatch" not in good.render()

    bad = good.model_copy(update={"taxonomy_consistent": False})
    assert "mismatch" in bad.render()


def test_training_config_defaults_are_sane() -> None:
    config = TrainingConfig()
    assert config.C > 0
    assert config.class_weight is None  # preserves calibration
    assert isinstance(config.features, FeatureConfig)


def test_training_config_rejects_bad_values() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(C=0)
    with pytest.raises(ValidationError):
        TrainingConfig(features=FeatureConfig(max_df=1.5))


def test_model_metadata_render_lists_label_space() -> None:
    meta = ModelMetadata(
        model_version="0.1.0",
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
        python_version="3.14.0",
        sklearn_version="1.8.0",
        numpy_version="2.4.0",
        train_dataset_name="nirnayax-train",
        train_dataset_split="train",
        train_size=400,
        train_dataset_seed=20260101,
        train_fingerprint="deadbeefdeadbeef",
        targets=("category", "subcategory", "priority"),
        label_space={
            "category": ("NETWORK", "APPLICATION_DB"),
            "subcategory": ("BGP_ROUTING",),
            "priority": ("P1", "P2", "P3", "P4"),
        },
        training_config=TrainingConfig(),
    )
    rendered = meta.render()
    assert "0.1.0" in rendered
    assert "category=2" in rendered
    assert "priority=4" in rendered
