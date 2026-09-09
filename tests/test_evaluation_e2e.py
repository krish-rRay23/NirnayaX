"""Tests for End-to-End Evaluation module."""

from __future__ import annotations

from nirnayax.data import generate_dataset
from nirnayax.evaluation import E2EEvaluator
from nirnayax.ml import train_triage_model


def test_e2e_evaluator_runs_and_produces_report() -> None:
    dataset = generate_dataset(20, seed=42)
    model = train_triage_model(dataset)

    evaluator = E2EEvaluator(model, dataset)
    report = evaluator.evaluate(sample_size=10)

    assert report.num_incidents == 10
    assert 0.0 <= report.ml_category_acc <= 1.0
    assert 0.0 <= report.retrieval_recall_at_3 <= 1.0
    assert report.triage_mean_ms >= 0.0
    rendered = report.render()
    assert "NirnayaX End-to-End Evaluation Report" in rendered
