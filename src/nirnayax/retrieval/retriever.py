"""Hybrid retrieval: dense + lexical, fused, filtered, reranked, and cited.

:class:`HybridRetriever` is the reusable core the future agent talks to. It owns a
chunk corpus with a dense :class:`~nirnayax.retrieval.vector.VectorIndex` and a
lexical :class:`~nirnayax.retrieval.lexical.Bm25Index`, fuses their rankings with
Reciprocal Rank Fusion, optionally reranks the top candidates, applies metadata
filters, and returns cited :class:`~nirnayax.retrieval.types.RetrievalResult`s.

Convenience builders adapt the two Phase-1 record types:

* :func:`build_runbook_retriever` — the runbook knowledge base (a document store).
* :func:`build_incident_retriever` — historical incidents (for correlation), each
  kept as a single chunk so results map one-to-one to incidents.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from ..domain.models import Incident, Runbook
from .chunking import (
    chunk_documents,
    document_from_incident,
    document_from_runbook,
)
from .embeddings import Embedder
from .lexical import Bm25Index
from .ranking import LexicalReranker, Reranker, reciprocal_rank_fusion
from .types import Chunk, ChunkConfig, Document, MetaValue, RetrievalConfig, RetrievalResult
from .vector import VectorIndex

#: A metadata filter: field -> required scalar, or a set of acceptable scalars.
FilterValue = MetaValue | frozenset[MetaValue] | set[MetaValue]
Filters = Mapping[str, FilterValue]

#: Large enough that a single incident record is never split into two chunks.
_INCIDENT_CHUNK = ChunkConfig(max_chars=8000, overlap_chars=0, min_chunk_chars=0)


def _matches(value: MetaValue | None, expected: FilterValue) -> bool:
    """True if a chunk's metadata ``value`` satisfies one filter ``expected``."""

    if isinstance(expected, set | frozenset):
        if isinstance(value, tuple):  # e.g. tags: any overlap with the accepted set
            return bool(set(value) & expected)
        return value in expected
    if isinstance(value, tuple):  # scalar filter against a tuple => membership
        return expected in value
    return value == expected


def match_metadata(metadata: Mapping[str, MetaValue], filters: Filters) -> bool:
    """True if ``metadata`` satisfies every ``field -> constraint`` in ``filters``."""

    return all(_matches(metadata.get(field), expected) for field, expected in filters.items())


class HybridRetriever:
    """Dense + lexical hybrid retriever over a fixed chunk corpus."""

    def __init__(
        self,
        *,
        chunks: Sequence[Chunk],
        vector_index: VectorIndex,
        bm25_index: Bm25Index,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self._vector = vector_index
        self._bm25 = bm25_index
        self._reranker = reranker
        self._config = config or RetrievalConfig()

    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        *,
        embedder: Embedder | None = None,
        chunk_config: ChunkConfig | None = None,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
    ) -> HybridRetriever:
        """Chunk ``documents``, build both indexes, and assemble a retriever."""

        chunks = chunk_documents(documents, chunk_config)
        if not chunks:
            raise ValueError("no chunks produced from the provided documents")
        texts = [c.text for c in chunks]
        vector_index = VectorIndex.build(texts, embedder)
        bm25_index = Bm25Index.build(texts)
        return cls(
            chunks=chunks,
            vector_index=vector_index,
            bm25_index=bm25_index,
            reranker=reranker,
            config=config,
        )

    @property
    def config(self) -> RetrievalConfig:
        return self._config

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    def _allowed_indices(self, filters: Filters | None) -> set[int] | None:
        if not filters:
            return None
        return {i for i, c in enumerate(self._chunks) if match_metadata(c.metadata, filters)}

    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: Filters | None = None,
        exclude_source_ids: Collection[str] | None = None,
        use_reranker: bool | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top-``k`` chunks for ``query``.

        Pipeline: dense + lexical ranking → metadata filtering → RRF fusion →
        (optional) rerank of the top ``candidate_k`` → truncate to ``k``. Each
        result carries its component scores and a citation.
        """

        k = k or self._config.top_k
        rerank = self._config.use_reranker if use_reranker is None else use_reranker
        allowed = self._allowed_indices(filters)
        exclude = set(exclude_source_ids or ())

        vec_ranked = self._vector.search(query)
        lex_ranked = self._bm25.search(query)

        def keep(idx: int) -> bool:
            if allowed is not None and idx not in allowed:
                return False
            return self._chunks[idx].source_id not in exclude

        vec_ranked = [(i, s) for i, s in vec_ranked if keep(i)]
        lex_ranked = [(i, s) for i, s in lex_ranked if keep(i)]
        vec_scores = dict(vec_ranked)
        lex_scores = dict(lex_ranked)

        fused = reciprocal_rank_fusion(
            [[i for i, _ in vec_ranked], [i for i, _ in lex_ranked]],
            k=self._config.rrf_k,
        )
        fused = fused[: self._config.candidate_k]
        fused_scores = dict(fused)
        order = [i for i, _ in fused]

        rerank_scores: dict[int, float] = {}
        if rerank and self._reranker is not None and order:
            candidates = [self._chunks[i] for i in order]
            reranked = self._reranker.rerank(query, candidates)
            rerank_scores = {order[local]: score for local, score in reranked}
            order = [order[local] for local, _ in reranked]

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(order[:k], start=1):
            components = {
                "vector": vec_scores.get(idx, 0.0),
                "bm25": lex_scores.get(idx, 0.0),
                "rrf": fused_scores.get(idx, 0.0),
            }
            if idx in rerank_scores:
                components["rerank"] = rerank_scores[idx]
            score = rerank_scores[idx] if idx in rerank_scores else fused_scores.get(idx, 0.0)
            results.append(
                RetrievalResult(
                    chunk=self._chunks[idx],
                    score=score,
                    rank=rank,
                    components=components,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Convenience builders for the Phase-1 record types
# ---------------------------------------------------------------------------
def build_runbook_retriever(
    runbooks: Sequence[Runbook],
    *,
    config: RetrievalConfig | None = None,
    chunk_config: ChunkConfig | None = None,
    reranker: Reranker | None = None,
    embedder: Embedder | None = None,
) -> HybridRetriever:
    """Build a retriever over the runbook knowledge base."""

    documents = [document_from_runbook(rb) for rb in runbooks]
    return HybridRetriever.build(
        documents,
        embedder=embedder,
        chunk_config=chunk_config,
        config=config,
        reranker=reranker or LexicalReranker(),
    )


def build_incident_retriever(
    incidents: Sequence[Incident],
    *,
    config: RetrievalConfig | None = None,
    reranker: Reranker | None = None,
    embedder: Embedder | None = None,
) -> HybridRetriever:
    """Build a retriever over historical incidents (one chunk per incident)."""

    documents = [document_from_incident(inc) for inc in incidents]
    return HybridRetriever.build(
        documents,
        embedder=embedder,
        chunk_config=_INCIDENT_CHUNK,
        config=config,
        reranker=reranker or LexicalReranker(),
    )


def similar_incidents(
    retriever: HybridRetriever,
    incident: Incident,
    *,
    k: int = 5,
    filters: Filters | None = None,
    use_reranker: bool | None = None,
) -> list[RetrievalResult]:
    """Find the ``k`` most similar historical incidents to ``incident``.

    Queries with the incident's own text and excludes itself from the results, so
    it works whether or not ``incident`` is part of the indexed corpus.
    """

    query = document_from_incident(incident).text
    return retriever.retrieve(
        query,
        k=k,
        filters=filters,
        exclude_source_ids={incident.incident_id},
        use_reranker=use_reranker,
    )


__all__ = [
    "Filters",
    "HybridRetriever",
    "build_incident_retriever",
    "build_runbook_retriever",
    "match_metadata",
    "similar_incidents",
]
