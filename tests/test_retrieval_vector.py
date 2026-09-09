"""Tests for the dense vector index."""

from __future__ import annotations

import pytest

from nirnayax.retrieval.vector import VectorIndex

_CORPUS = [
    "bgp session flapping between core routers",
    "database connection pool exhausted clients timeout",
    "dns resolution failing nxdomain across region",
    "bgp peering neighbor stuck active prefixes withdrawn",
]


def test_vector_search_ranks_relevant_first() -> None:
    index = VectorIndex.build(_CORPUS)
    ranked = index.search("database connection pool timeout")
    assert ranked[0][0] == 1
    assert ranked[0][1] > ranked[-1][1]


def test_vector_search_k_and_all() -> None:
    index = VectorIndex.build(_CORPUS)
    assert len(index.search("bgp", k=2)) == 2
    assert len(index.search("bgp")) == len(_CORPUS) == len(index)


def test_vector_search_ties_break_by_index() -> None:
    index = VectorIndex.build(["alpha beta", "alpha beta", "gamma delta"])
    ranked = index.search("alpha beta")
    assert [idx for idx, _ in ranked[:2]] == [0, 1]


def test_vector_build_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        VectorIndex.build([])
