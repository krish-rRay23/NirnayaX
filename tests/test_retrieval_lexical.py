"""Tests for the Okapi BM25 lexical index."""

from __future__ import annotations

from nirnayax.retrieval.lexical import Bm25Index

_CORPUS = [
    "bgp session flapping between core routers, routes withdrawn",
    "database connection pool exhausted, clients time out waiting",
    "dns resolution failing with nxdomain across the region",
    "bgp peering neighbor stuck in active state, prefixes vanish",
]


def test_bm25_ranks_relevant_document_first() -> None:
    index = Bm25Index.build(_CORPUS)
    ranked = index.search("database connection pool")
    assert ranked[0][0] == 1  # the DB pool document
    assert ranked[0][1] > 0.0


def test_bm25_more_query_terms_scores_higher() -> None:
    index = Bm25Index.build(_CORPUS)
    # Both BGP docs match "bgp"; doc 0 also has "routes withdrawn".
    ranked = dict(index.search("bgp routes withdrawn"))
    assert ranked[0] > ranked[3] > 0.0


def test_bm25_k_truncates_and_none_returns_all() -> None:
    index = Bm25Index.build(_CORPUS)
    assert len(index.search("bgp", k=2)) == 2
    assert len(index.search("bgp")) == len(_CORPUS) == len(index)


def test_bm25_no_overlap_scores_zero_and_orders_by_index() -> None:
    index = Bm25Index.build(_CORPUS)
    ranked = index.search("kubernetes helm chart")  # no shared vocabulary
    assert all(score == 0.0 for _, score in ranked)
    assert [idx for idx, _ in ranked] == [0, 1, 2, 3]


def test_bm25_ties_break_by_ascending_index() -> None:
    index = Bm25Index.build(["same words here", "same words here", "unrelated"])
    ranked = index.search("same words")
    assert [idx for idx, _ in ranked[:2]] == [0, 1]


def test_bm25_is_deterministic() -> None:
    a = Bm25Index.build(_CORPUS).search("bgp session")
    b = Bm25Index.build(_CORPUS).search("bgp session")
    assert a == b
