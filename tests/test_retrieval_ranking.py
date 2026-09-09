"""Tests for Reciprocal Rank Fusion and the lexical reranker."""

from __future__ import annotations

import pytest

from nirnayax.retrieval.ranking import LexicalReranker, Reranker, reciprocal_rank_fusion
from nirnayax.retrieval.types import Chunk


def _chunk(index: int, text: str, *, title: str = "") -> Chunk:
    return Chunk(
        chunk_id=f"D#{index}",
        doc_id="D",
        index=index,
        text=text,
        start=0,
        end=len(text),
        source_type="test",
        source_id="D",
        title=title,
    )


def test_rrf_combines_rank_positions() -> None:
    # doc 0 is high in both lists; doc 2 tops list b; doc 1 is low in both.
    fused = reciprocal_rank_fusion([[0, 1, 2], [2, 0, 1]], k=60)
    assert [idx for idx, _ in fused] == [0, 2, 1]
    assert fused[0][1] > fused[1][1] > fused[2][1]


def test_rrf_weights_favor_the_heavier_list() -> None:
    assert reciprocal_rank_fusion([[0, 1], [1, 0]], weights=[1.0, 5.0])[0][0] == 1
    assert reciprocal_rank_fusion([[0, 1], [1, 0]], weights=[5.0, 1.0])[0][0] == 0


def test_rrf_mismatched_weights_raise() -> None:
    with pytest.raises(ValueError, match="weights must align"):
        reciprocal_rank_fusion([[0, 1]], weights=[1.0, 2.0])


def test_reranker_orders_by_token_overlap() -> None:
    reranker = LexicalReranker()
    candidates = [
        _chunk(0, "database connection pool exhausted"),
        _chunk(1, "bgp session flapping routes withdrawn"),
    ]
    ranked = reranker.rerank("bgp session flapping between routers", candidates)
    assert ranked[0][0] == 1  # the BGP chunk
    assert ranked[0][1] > ranked[1][1]


def test_reranker_empty_query_is_stable_and_zero() -> None:
    reranker = LexicalReranker()
    candidates = [_chunk(0, "alpha beta"), _chunk(1, "gamma delta")]
    ranked = reranker.rerank("", candidates)
    assert [idx for idx, _ in ranked] == [0, 1]
    assert all(score == 0.0 for _, score in ranked)


def test_reranker_title_boost_breaks_a_tie() -> None:
    reranker = LexicalReranker()
    candidates = [
        _chunk(0, "alpha beta gamma"),
        _chunk(1, "alpha beta gamma", title="bgp"),
    ]
    ranked = reranker.rerank("alpha bgp", candidates)
    assert ranked[0][0] == 1  # equal body overlap, but title matches the query


def test_lexical_reranker_satisfies_protocol() -> None:
    assert isinstance(LexicalReranker(), Reranker)
