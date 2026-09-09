"""Dense vector search over an embedded corpus.

A thin wrapper around an :class:`~nirnayax.retrieval.embeddings.Embedder`: it
embeds the corpus once at build time, then ranks documents against a query by
cosine similarity. Deterministic; ties break by ascending document index (via a
stable argsort), matching the lexical index.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .embeddings import Embedder, EmbeddingMatrix, TfidfEmbedder, cosine_similarity


class VectorIndex:
    """A dense embedding index supporting cosine top-k search."""

    def __init__(self, matrix: EmbeddingMatrix, embedder: Embedder) -> None:
        self._matrix = matrix
        self._embedder = embedder

    @classmethod
    def build(cls, documents: Sequence[str], embedder: Embedder | None = None) -> VectorIndex:
        """Embed ``documents`` and build an index.

        If no embedder is supplied, a :class:`TfidfEmbedder` is fitted on the
        documents themselves — the default offline configuration.
        """

        if not documents:
            raise ValueError("cannot build a vector index over an empty corpus")
        embedder = embedder or TfidfEmbedder.fit(documents)
        matrix = embedder.embed(documents)
        return cls(matrix, embedder)

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def __len__(self) -> int:
        return int(self._matrix.shape[0])

    def search(self, query: str, k: int | None = None) -> list[tuple[int, float]]:
        """Rank all documents by cosine similarity to ``query`` (descending).

        Returns ``(doc_index, score)`` pairs; ``k=None`` returns every document.
        """

        query_vec = self._embedder.embed([query])[0]
        scores = cosine_similarity(self._matrix, query_vec)
        # Stable argsort on the negated scores => descending, ties by index asc.
        order = np.argsort(-scores, kind="stable")
        ranked = [(int(i), float(scores[i])) for i in order]
        return ranked if k is None else ranked[:k]


__all__ = ["VectorIndex"]
