"""Unit + integration tests for inference and persistence.

Covers the inference contract (calibrated confidences in ``[0, 1]``, a full
probability distribution that sums to one, taxonomy consistency) and a
save/load round-trip that must preserve predictions exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nirnayax.domain.models import IncidentDataset
from nirnayax.ml import TARGETS, TicketDraft, TriageModel

_TOL = 1e-9


def test_predict_returns_all_three_heads(
    trained_model: TriageModel, sample_draft: TicketDraft
) -> None:
    pred = trained_model.predict(sample_draft)
    assert pred.model_version == trained_model.model_version
    for head in (pred.category, pred.subcategory, pred.priority):
        assert 0.0 <= head.confidence <= 1.0
        assert head.label == head.distribution[0][0]
        assert head.confidence == pytest.approx(head.distribution[0][1])


def test_distributions_are_valid_probabilities(
    trained_model: TriageModel, sample_draft: TicketDraft
) -> None:
    pred = trained_model.predict(sample_draft)
    expected_sizes = {"category": 4, "subcategory": 17, "priority": 4}
    for name, head in (
        ("category", pred.category),
        ("subcategory", pred.subcategory),
        ("priority", pred.priority),
    ):
        probs = [p for _, p in head.distribution]
        assert len(head.distribution) == expected_sizes[name]
        assert sum(probs) == pytest.approx(1.0, abs=_TOL)
        # Sorted by descending probability.
        assert probs == sorted(probs, reverse=True)
        assert all(0.0 <= p <= 1.0 for p in probs)


def test_clear_network_ticket_is_taxonomy_consistent(
    trained_model: TriageModel, sample_draft: TicketDraft
) -> None:
    pred = trained_model.predict(sample_draft)
    assert pred.category.label == "NETWORK"
    assert pred.taxonomy_consistent is True


def test_classes_for_matches_label_space(trained_model: TriageModel) -> None:
    assert len(trained_model.classes_for("category")) == 4
    assert len(trained_model.classes_for("subcategory")) == 17
    assert len(trained_model.classes_for("priority")) == 4


def test_predict_incident_recovers_labels_for_training_data(
    trained_model: TriageModel, train_dataset: IncidentDataset
) -> None:
    """On separable synthetic data the model should recover category/subcategory."""

    incidents = train_dataset.incidents[:50]
    cat_hits = sum(
        trained_model.predict_incident(inc).category.label == inc.category.value
        for inc in incidents
    )
    assert cat_hits / len(incidents) >= 0.9


def test_score_incidents_shapes(trained_model: TriageModel, eval_dataset: IncidentDataset) -> None:
    incidents = eval_dataset.incidents
    scores = trained_model.score_incidents(incidents)
    assert set(scores) == set(TARGETS)
    for target in TARGETS:
        classes, proba = scores[target]
        assert proba.shape == (len(incidents), len(classes))


def test_save_load_round_trip_preserves_predictions(
    trained_model: TriageModel, sample_draft: TicketDraft, tmp_path: Path
) -> None:
    path = tmp_path / "triage.joblib"
    returned = trained_model.save(path)
    assert returned == path
    assert path.exists()

    meta_path = tmp_path / "triage.joblib.meta.json"
    assert meta_path.exists()
    sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
    assert sidecar["model_version"] == trained_model.model_version

    reloaded = TriageModel.load(path)
    assert reloaded.metadata.model_dump(mode="json") == trained_model.metadata.model_dump(
        mode="json"
    )
    assert reloaded.predict(sample_draft) == trained_model.predict(sample_draft)


def test_missing_head_is_rejected(trained_model: TriageModel) -> None:
    with pytest.raises(ValueError, match="missing classifier head"):
        TriageModel(
            vectorizer=object(),
            classifiers={"category": object()},  # subcategory + priority missing
            metadata=trained_model.metadata,
        )


def test_load_rejects_metadata_missing_from_payload(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "broken.joblib"
    joblib.dump({"vectorizer": object(), "classifiers": {}}, path)
    with pytest.raises(KeyError):
        TriageModel.load(path)
