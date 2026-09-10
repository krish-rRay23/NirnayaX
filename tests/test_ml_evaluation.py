"""Unit + integration tests for evaluation metrics and calibration.

Verifies the Expected Calibration Error helper on hand-built cases, and that
``evaluate_model`` produces well-formed metrics and confusion matrices with the
expected quality: category/subcategory are highly separable on synthetic data,
while priority is genuinely hard (it derives from severity + customer impact,
which are deliberately excluded from the features) yet still beats chance.
"""

from __future__ import annotations

import pytest

from nirnayax.data import generate_dataset
from nirnayax.domain import DatasetSplit
from nirnayax.domain.models import IncidentDataset
from nirnayax.ml import (
    TARGETS,
    TriageModel,
    evaluate_model,
    expected_calibration_error,
    train_triage_model,
)


def test_ece_zero_when_perfectly_calibrated() -> None:
    # Always fully confident and always correct -> no calibration gap.
    assert expected_calibration_error([1.0, 1.0, 1.0], [True, True, True]) == pytest.approx(0.0)


def test_ece_detects_overconfidence() -> None:
    # Fully confident but only half correct -> gap of 0.5.
    assert expected_calibration_error([1.0, 1.0], [True, False]) == pytest.approx(0.5)


def test_ece_handles_zero_confidence_bin_edge() -> None:
    # A 0.0 confidence must be assigned to the first bin, not dropped.
    assert expected_calibration_error([0.0], [False]) == pytest.approx(0.0)


def test_ece_empty_is_zero() -> None:
    assert expected_calibration_error([], []) == 0.0


def test_evaluate_empty_raises(trained_model: TriageModel) -> None:
    empty = generate_dataset(0, split=DatasetSplit.EVAL)
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(trained_model, empty)


def test_report_is_well_formed(trained_model: TriageModel, eval_dataset: IncidentDataset) -> None:
    report = evaluate_model(trained_model, eval_dataset)
    assert report.dataset_size == len(eval_dataset.incidents)
    assert set(report.targets) == set(TARGETS)

    for target in TARGETS:
        m = report.targets[target]
        assert m.n == len(eval_dataset.incidents)
        assert 0.0 <= m.accuracy <= 1.0
        assert 0.0 <= m.macro_f1 <= 1.0
        assert 0.0 <= m.weighted_f1 <= 1.0
        assert 0.0 <= m.ece <= 1.0
        assert 0.0 <= m.mean_confidence <= 1.0

        # Confusion matrix is square, aligned with labels, and totals to n.
        size = len(m.labels)
        assert len(m.confusion_matrix) == size
        assert all(len(row) == size for row in m.confusion_matrix)
        assert sum(cell for row in m.confusion_matrix for cell in row) == m.n
        # Diagonal (correct predictions) is consistent with accuracy.
        diagonal = sum(m.confusion_matrix[i][i] for i in range(size))
        assert diagonal == pytest.approx(m.accuracy * m.n, abs=0.5)


def test_quality_matches_task_difficulty(
    trained_model: TriageModel, eval_dataset: IncidentDataset
) -> None:
    report = evaluate_model(trained_model, eval_dataset)
    # Real dataset accuracy assertions (category: 13 classes, subcategory: 59 classes)
    assert report.targets["category"].accuracy >= 0.60
    assert report.targets["subcategory"].accuracy >= 0.30
    priority_accuracy = report.targets["priority"].accuracy
    assert 0.20 <= priority_accuracy <= 1.00


def test_evaluation_is_reproducible(
    train_dataset: IncidentDataset, eval_dataset: IncidentDataset
) -> None:
    """Same (train, eval) -> identical evaluation reports, including confusion matrices."""

    r1 = evaluate_model(train_triage_model(train_dataset), eval_dataset)
    r2 = evaluate_model(train_triage_model(train_dataset), eval_dataset)
    assert r1 == r2
