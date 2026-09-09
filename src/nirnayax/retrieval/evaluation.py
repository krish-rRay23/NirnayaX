"""Retrieval evaluation: Recall@K, Precision@K, and MRR against gold labels.

The Phase-1 data provides honest relevance labels for free:

* **Runbook retrieval** — each subcategory maps to exactly one runbook, so an
  incident's single relevant document is the runbook for its subcategory. Here
  Recall@K is a hit-rate and MRR is the headline.
* **Incident correlation** — the relevant neighbours of an incident are the other
  incidents sharing its subcategory. With many relevant items, Precision@K and MRR
  are the meaningful measures; Recall@K is reported too, with the honest
  ``/|relevant|`` denominator.

Chunk-level results are collapsed to unique source ids (a runbook / incident) in
rank order before scoring, so multiple chunks of one source count once.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.models import Incident, Runbook
from .chunking import document_from_incident
from .retriever import HybridRetriever, similar_incidents
from .types import RetrievalMetrics, RetrievalResult

_DEFAULT_KS = (1, 3, 5)


def ranked_source_ids(
    results: Sequence[RetrievalResult], *, exclude: str | None = None
) -> list[str]:
    """Collapse ranked results to unique ``source_id``s, preserving order."""

    seen: list[str] = []
    for result in results:
        sid = result.chunk.source_id
        if sid == exclude or sid in seen:
            continue
        seen.append(sid)
    return seen


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of all relevant ids that appear in the top ``k`` (0 if none relevant)."""

    if not relevant:
        return 0.0
    hits = sum(1 for sid in ranked[:k] if sid in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of the top ``k`` retrieved ids that are relevant."""

    if k <= 0:
        return 0.0
    hits = sum(1 for sid in ranked[:k] if sid in relevant)
    return hits / k


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    """Reciprocal of the rank of the first relevant id (0 if none present)."""

    for rank, sid in enumerate(ranked, start=1):
        if sid in relevant:
            return 1.0 / rank
    return 0.0


def _aggregate(
    name: str,
    per_query_ranked: Sequence[tuple[list[str], frozenset[str]]],
    k_values: Sequence[int],
) -> RetrievalMetrics:
    """Average per-query Recall@K / Precision@K / RR into a metrics report."""

    n = len(per_query_ranked)
    recall = dict.fromkeys(k_values, 0.0)
    precision = dict.fromkeys(k_values, 0.0)
    mrr = 0.0
    for ranked, relevant in per_query_ranked:
        for k in k_values:
            recall[k] += recall_at_k(ranked, relevant, k)
            precision[k] += precision_at_k(ranked, relevant, k)
        mrr += reciprocal_rank(ranked, relevant)
    if n:
        recall = {k: v / n for k, v in recall.items()}
        precision = {k: v / n for k, v in precision.items()}
        mrr /= n
    return RetrievalMetrics(
        name=name,
        n_queries=n,
        k_values=tuple(k_values),
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr,
    )


def evaluate_runbook_retrieval(
    retriever: HybridRetriever,
    incidents: Sequence[Incident],
    runbooks: Sequence[Runbook],
    *,
    k_values: Sequence[int] = _DEFAULT_KS,
    use_reranker: bool | None = None,
) -> RetrievalMetrics:
    """Evaluate runbook retrieval: query = incident, gold = its subcategory's runbook."""

    gold = {rb.subcategory.value: rb.runbook_id for rb in runbooks}
    depth = max(retriever.config.candidate_k, max(k_values))
    per_query: list[tuple[list[str], frozenset[str]]] = []
    for incident in incidents:
        relevant = gold.get(incident.subcategory.value)
        if relevant is None:
            continue
        query = document_from_incident(incident).text
        results = retriever.retrieve(query, k=depth, use_reranker=use_reranker)
        per_query.append((ranked_source_ids(results), frozenset({relevant})))
    return _aggregate("runbook", per_query, k_values)


def evaluate_incident_retrieval(
    retriever: HybridRetriever,
    query_incidents: Sequence[Incident],
    corpus_incidents: Sequence[Incident],
    *,
    k_values: Sequence[int] = _DEFAULT_KS,
    use_reranker: bool | None = None,
) -> RetrievalMetrics:
    """Evaluate incident correlation: relevant = same-subcategory corpus incidents.

    ``retriever`` must index ``corpus_incidents``. Each query excludes itself, so
    the query and corpus sets may overlap (leave-one-out) or be disjoint.
    """

    depth = max(retriever.config.candidate_k, max(k_values))
    by_subcat: dict[str, set[str]] = {}
    for inc in corpus_incidents:
        by_subcat.setdefault(inc.subcategory.value, set()).add(inc.incident_id)

    per_query: list[tuple[list[str], frozenset[str]]] = []
    for incident in query_incidents:
        relevant = by_subcat.get(incident.subcategory.value, set()) - {incident.incident_id}
        if not relevant:
            continue
        results = similar_incidents(retriever, incident, k=depth, use_reranker=use_reranker)
        ranked = ranked_source_ids(results, exclude=incident.incident_id)
        per_query.append((ranked, frozenset(relevant)))
    return _aggregate("incident-similarity", per_query, k_values)


__all__ = [
    "evaluate_incident_retrieval",
    "evaluate_runbook_retrieval",
    "precision_at_k",
    "ranked_source_ids",
    "recall_at_k",
    "reciprocal_rank",
]
