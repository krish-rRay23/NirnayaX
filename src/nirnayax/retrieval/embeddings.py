"""Embeddings: a provider-agnostic protocol and a default TF-IDF implementation.

The retrieval layer never hard-depends on any embedding provider. It talks to the
:class:`Embedder` protocol; a future agent can drop in an LLM / sentence-transformer
embedder by implementing the same three members. The default :class:`TfidfEmbedder`
reuses the scikit-learn stack already present for the ML engine, so no new heavy
dependency is introduced and everything stays fully deterministic and offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

#: Dense embedding matrix type (rows = documents, cols = embedding dimensions).
EmbeddingMatrix = NDArray[np.float64]


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to L2-normalized dense vectors (so dot product == cosine)."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Return an ``(len(texts), dim)`` float matrix of unit-norm rows."""
        ...


class TfidfEmbedder:
    """Default offline embedder: a fitted TF-IDF vectorizer with L2-normalized rows.

    Fit it once on the chunk corpus with :meth:`fit`; both corpus and query
    embeddings then come from the same vocabulary, so cosine similarity is
    meaningful. Rows are unit-normalized (``norm="l2"``), hence a dot product is
    the cosine similarity.
    """

    def __init__(self, vectorizer: Any, dim: int) -> None:
        self._vectorizer = vectorizer
        self._dim = dim

    @classmethod
    def fit(
        cls,
        corpus: Sequence[str],
        *,
        ngram_max: int = 2,
        min_df: int = 1,
        max_df: float = 1.0,
        use_stopwords: bool = True,
    ) -> TfidfEmbedder:
        """Fit a TF-IDF vectorizer on ``corpus`` and return an embedder.

        Defaults suit a small knowledge base (``min_df=1`` keeps rare-but-useful
        terms). Determinism follows from TF-IDF being a closed-form transform.
        """

        if not corpus:
            raise ValueError("cannot fit an embedder on an empty corpus")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, ngram_max),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            stop_words="english" if use_stopwords else None,
            norm="l2",
        )
        vectorizer.fit(list(corpus))
        return cls(vectorizer, len(vectorizer.vocabulary_))

    @property
    def name(self) -> str:
        return "tfidf"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        dense = self._vectorizer.transform(list(texts)).toarray()
        return np.asarray(dense, dtype=np.float64)


def cosine_similarity(matrix: EmbeddingMatrix, vector: EmbeddingMatrix) -> EmbeddingMatrix:
    """Cosine similarity of each row of ``matrix`` against ``vector``.

    Assumes both are already L2-normalized (as :class:`TfidfEmbedder` guarantees),
    so this is a plain dot product; zero rows/vectors yield 0.0.
    """

    scores: EmbeddingMatrix = matrix @ vector
    return scores


__all__ = ["Embedder", "EmbeddingMatrix", "TfidfEmbedder", "cosine_similarity"]
