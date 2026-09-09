"""Result fusion and reranking.

Two post-retrieval stages, kept separate from the indexes so both are trivially
testable and swappable:

* :func:`reciprocal_rank_fusion` — combine several ranked lists (dense + lexical)
  into one, using only rank positions, so incomparable score scales don't matter.
* :class:`Reranker` — a provider-agnostic protocol for a second-stage reorder. The
  default :class:`LexicalReranker` is a lightweight, dependency-free scorer; a
  future agent can drop in a cross-encoder / LLM reranker behind the same API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .chunking import tokenize
from .types import Chunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked id lists via Reciprocal Rank Fusion.

    Each input is a list of document indices, best first. An item's fused score is
    ``sum(weight / (k + rank))`` over the lists it appears in (rank 1-based). The
    result is sorted by descending fused score, ties broken by ascending index.
    """

    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must align with rankings")

    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@runtime_checkable
class Reranker(Protocol):
    """Reorders first-stage candidates for a query.

    ``rerank`` receives the query and the candidate chunks and returns
    ``(candidate_index, score)`` pairs — indices into the *input* sequence —
    sorted best first. It must be a pure function of its inputs (deterministic).
    """

    @property
    def name(self) -> str: ...

    def rerank(self, query: str, candidates: Sequence[Chunk]) -> list[tuple[int, float]]: ...


class LexicalReranker:
    """A dependency-free reranker scoring query/chunk token-overlap affinity.

    The score is the F1 of shared tokens between query and chunk text (a
    set-overlap signal distinct from BM25's length-normalized term weighting),
    plus a small boost when the chunk's title also overlaps the query. It is a
    deterministic, offline stand-in for a heavier cross-encoder.
    """

    def __init__(self, *, title_boost: float = 0.1) -> None:
        self._title_boost = title_boost

    @property
    def name(self) -> str:
        return "lexical-overlap"

    def _affinity(self, query_tokens: set[str], text: str) -> float:
        cand = set(tokenize(text))
        if not cand or not query_tokens:
            return 0.0
        overlap = len(query_tokens & cand)
        if overlap == 0:
            return 0.0
        precision = overlap / len(cand)
        recall = overlap / len(query_tokens)
        return 2.0 * precision * recall / (precision + recall)

    def rerank(self, query: str, candidates: Sequence[Chunk]) -> list[tuple[int, float]]:
        query_tokens = set(tokenize(query))
        scored: list[tuple[int, float]] = []
        for i, chunk in enumerate(candidates):
            score = self._affinity(query_tokens, chunk.text)
            if chunk.title and query_tokens & set(tokenize(chunk.title)):
                score += self._title_boost
            scored.append((i, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored


__all__ = ["LexicalReranker", "Reranker", "reciprocal_rank_fusion"]
