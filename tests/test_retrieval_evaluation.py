"""Tests for retrieval evaluation metrics and benchmark runners."""

from __future__ import annotations

import pytest

from nirnayax.domain.models import IncidentDataset, Runbook
from nirnayax.retrieval import HybridRetriever
from nirnayax.retrieval.evaluation import (
    evaluate_incident_retrieval,
    evaluate_runbook_retrieval,
    precision_at_k,
    ranked_source_ids,
    recall_at_k,
    reciprocal_rank,
)
from nirnayax.retrieval.types import Chunk, RetrievalMetrics, RetrievalResult


def _mock_result(source_id: str, rank: int) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=f"{source_id}#0",
        doc_id=source_id,
        index=0,
        text="text",
        start=0,
        end=4,
        source_type="runbook",
        source_id=source_id,
    )
    return RetrievalResult(chunk=chunk, score=1.0 / rank, rank=rank)


def test_ranked_source_ids_deduplicates_and_excludes() -> None:
    results = [
        _mock_result("RB-1", 1),
        _mock_result("RB-1", 2),
        _mock_result("RB-2", 3),
        _mock_result("RB-3", 4),
    ]
    assert ranked_source_ids(results) == ["RB-1", "RB-2", "RB-3"]
    assert ranked_source_ids(results, exclude="RB-1") == ["RB-2", "RB-3"]


def test_recall_at_k() -> None:
    relevant = frozenset({"A", "B"})
    assert recall_at_k(["A", "C", "D"], relevant, k=1) == pytest.approx(0.5)
    assert recall_at_k(["A", "B", "D"], relevant, k=2) == pytest.approx(1.0)
    assert recall_at_k(["X", "Y"], relevant, k=5) == 0.0
    assert recall_at_k(["A", "B"], frozenset(), k=5) == 0.0


def test_precision_at_k() -> None:
    relevant = frozenset({"A", "B"})
    assert precision_at_k(["A", "C", "D"], relevant, k=1) == pytest.approx(1.0)
    assert precision_at_k(["A", "C", "D"], relevant, k=2) == pytest.approx(0.5)
    assert precision_at_k(["X", "Y"], relevant, k=2) == 0.0
    assert precision_at_k(["A", "B"], relevant, k=0) == 0.0


def test_reciprocal_rank() -> None:
    relevant = frozenset({"B", "C"})
    assert reciprocal_rank(["A", "B", "C"], relevant) == pytest.approx(0.5)
    assert reciprocal_rank(["B", "A"], relevant) == pytest.approx(1.0)
    assert reciprocal_rank(["X", "Y"], relevant) == 0.0


def test_retrieval_metrics_render() -> None:
    metrics = RetrievalMetrics(
        name="test",
        n_queries=10,
        k_values=(1, 3, 5),
        recall_at_k={1: 0.5, 3: 0.8, 5: 0.9},
        precision_at_k={1: 0.5, 3: 0.3, 5: 0.2},
        mrr=0.75,
    )
    rendered = metrics.render()
    assert "Retrieval [test]" in rendered
    assert "n=10" in rendered
    assert "MRR=0.7500" in rendered
    assert "R@1=0.5000" in rendered
    assert "P@1=0.5000" in rendered


def test_evaluate_runbook_retrieval(
    runbook_retriever: HybridRetriever,
    eval_dataset: IncidentDataset,
    runbooks: tuple[Runbook, ...],
) -> None:
    metrics = evaluate_runbook_retrieval(
        runbook_retriever, eval_dataset.incidents, runbooks, k_values=(1, 3)
    )
    assert metrics.name == "runbook"
    assert metrics.n_queries == len(eval_dataset.incidents)
    assert metrics.k_values == (1, 3)
    assert 0.0 <= metrics.mrr <= 1.0
    assert 0.0 <= metrics.recall_at_k[1] <= 1.0


def test_evaluate_incident_retrieval(
    incident_retriever: HybridRetriever,
    eval_dataset: IncidentDataset,
    train_dataset: IncidentDataset,
) -> None:
    metrics = evaluate_incident_retrieval(
        incident_retriever, eval_dataset.incidents, train_dataset.incidents, k_values=(1, 3)
    )
    assert metrics.name == "incident-similarity"
    assert metrics.n_queries > 0
    assert metrics.k_values == (1, 3)
    assert 0.0 <= metrics.mrr <= 1.0
    assert 0.0 <= metrics.precision_at_k[1] <= 1.0
