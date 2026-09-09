"""Tests for the hybrid retriever, metadata filtering, and incident correlation."""

from __future__ import annotations

import pytest

from nirnayax.domain.models import IncidentDataset
from nirnayax.retrieval import HybridRetriever
from nirnayax.retrieval.retriever import (
    build_runbook_retriever,
    match_metadata,
    similar_incidents,
)

_BGP_QUERY = "BGP peering session keeps flapping and routes are being withdrawn"


# --- metadata filtering ----------------------------------------------------
def test_match_metadata_scalar_equality() -> None:
    assert match_metadata({"category": "NETWORK"}, {"category": "NETWORK"})
    assert not match_metadata({"category": "NETWORK"}, {"category": "BILLING_OSS"})


def test_match_metadata_missing_key_fails() -> None:
    assert not match_metadata({}, {"category": "NETWORK"})


def test_match_metadata_set_any_of() -> None:
    meta = {"subcategory": "BGP_ROUTING"}
    assert match_metadata(meta, {"subcategory": {"BGP_ROUTING", "LINK_DOWN"}})
    assert not match_metadata(meta, {"subcategory": {"DNS_RESOLUTION"}})


def test_match_metadata_tag_membership_and_overlap() -> None:
    meta = {"tags": ("peering", "bgp")}
    assert match_metadata(meta, {"tags": "bgp"})  # scalar against a tuple => membership
    assert not match_metadata(meta, {"tags": "dns"})
    assert match_metadata(meta, {"tags": {"bgp", "dns"}})  # set => any overlap
    assert not match_metadata(meta, {"tags": {"dns", "voip"}})


# --- hybrid retrieval ------------------------------------------------------
def test_retrieve_ranks_correct_runbook_first(runbook_retriever: HybridRetriever) -> None:
    results = runbook_retriever.retrieve(_BGP_QUERY, k=3)
    assert results
    assert results[0].chunk.metadata["subcategory"] == "BGP_ROUTING"


def test_results_are_well_formed(runbook_retriever: HybridRetriever) -> None:
    results = runbook_retriever.retrieve(_BGP_QUERY, k=5)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    for result in results:
        # Citation points back at the chunk it came from.
        assert result.citation.source_id == result.chunk.source_id
        assert result.citation.chunk_id == result.chunk.chunk_id
        # Component scores are recorded for transparency.
        assert {"vector", "bm25", "rrf"} <= set(result.components)


def test_metadata_filter_restricts_results(runbook_retriever: HybridRetriever) -> None:
    results = runbook_retriever.retrieve(
        "service is degraded", k=5, filters={"category": "NETWORK"}
    )
    assert results
    assert all(r.chunk.metadata["category"] == "NETWORK" for r in results)


def test_exclude_source_ids_drops_a_source(runbook_retriever: HybridRetriever) -> None:
    top = runbook_retriever.retrieve(_BGP_QUERY, k=1)[0].chunk.source_id
    again = runbook_retriever.retrieve(_BGP_QUERY, k=5, exclude_source_ids={top})
    assert all(r.chunk.source_id != top for r in again)


def test_retrieval_is_deterministic(runbook_retriever: HybridRetriever) -> None:
    a = runbook_retriever.retrieve(_BGP_QUERY, k=5)
    b = runbook_retriever.retrieve(_BGP_QUERY, k=5)
    assert [r.chunk.chunk_id for r in a] == [r.chunk.chunk_id for r in b]
    assert [r.score for r in a] == [r.score for r in b]


def test_reranker_toggle_still_returns_results(runbook_retriever: HybridRetriever) -> None:
    with_rr = runbook_retriever.retrieve(_BGP_QUERY, k=3, use_reranker=True)
    without_rr = runbook_retriever.retrieve(_BGP_QUERY, k=3, use_reranker=False)
    assert with_rr and without_rr
    # Both must still surface the correct subcategory at the top.
    assert with_rr[0].chunk.metadata["subcategory"] == "BGP_ROUTING"
    assert without_rr[0].chunk.metadata["subcategory"] == "BGP_ROUTING"


def test_build_empty_documents_raises() -> None:
    with pytest.raises(ValueError, match="no chunks"):
        build_runbook_retriever([])


# --- incident correlation --------------------------------------------------
def test_similar_incidents_excludes_self_and_matches_subcategory(
    incident_retriever: HybridRetriever, train_dataset: IncidentDataset
) -> None:
    query = train_dataset.incidents[0]
    results = similar_incidents(incident_retriever, query, k=5)

    assert results
    assert all(r.chunk.source_id != query.incident_id for r in results)
    # The nearest historical incident shares the query's subcategory.
    assert results[0].chunk.metadata["subcategory"] == query.subcategory.value


def test_citation_and_result_rendering(runbook_retriever: HybridRetriever) -> None:
    results = runbook_retriever.retrieve(_BGP_QUERY, k=1)
    res = results[0]
    rendered_cit = res.citation.render()
    rendered_res = res.render(width=30)

    assert res.chunk.source_id in rendered_cit
    assert f"#{res.rank}" in rendered_res
    assert "…" in rendered_res  # Truncated because width=30

