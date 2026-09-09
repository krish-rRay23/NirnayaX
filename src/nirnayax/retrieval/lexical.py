"""Okapi BM25 keyword search — a dependency-free lexical retriever.

Pure Python (no numpy / sklearn): the corpora here are small and BM25 is cheap,
so keeping it dependency-free lets the lexical half of hybrid retrieval be tested
in isolation. Scoring is deterministic; ties break by ascending document index.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from .chunking import tokenize


class Bm25Index:
    """An Okapi BM25 index over a fixed corpus of documents.

    ``k1`` controls term-frequency saturation and ``b`` controls length
    normalization (the standard defaults, 1.5 / 0.75, are used).
    """

    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._tfs: list[Counter[str]] = [Counter(tokenize(doc)) for doc in documents]
        self._doc_len: list[int] = [sum(tf.values()) for tf in self._tfs]
        n_docs = len(self._tfs)
        self._avgdl = (sum(self._doc_len) / n_docs) if n_docs else 0.0

        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        # BM25 idf with the +1 shift so every idf stays strictly positive.
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    @classmethod
    def build(cls, documents: Sequence[str], **kwargs: float) -> Bm25Index:
        """Alias for the constructor, matching the other indexes' ``build`` API."""

        return cls(documents, **kwargs)

    def __len__(self) -> int:
        return len(self._tfs)

    def score(self, query: str, doc_index: int) -> float:
        """BM25 score of a single document against ``query``."""

        tf = self._tfs[doc_index]
        if not tf:
            return 0.0
        denom_len = self._k1 * (1.0 - self._b + self._b * self._doc_len[doc_index] / self._avgdl)
        total = 0.0
        for term in tokenize(query):
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = self._idf.get(term, 0.0)
            total += idf * (freq * (self._k1 + 1.0)) / (freq + denom_len)
        return total

    def search(self, query: str, k: int | None = None) -> list[tuple[int, float]]:
        """Rank all documents against ``query`` (descending; ties by index).

        Returns ``(doc_index, score)`` pairs. ``k=None`` returns every document,
        which the hybrid retriever relies on for metadata filtering and fusion.
        """

        query_terms = set(tokenize(query))
        scored = [(idx, self._score_terms(query_terms, idx)) for idx in range(len(self._tfs))]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored if k is None else scored[:k]

    def _score_terms(self, query_terms: set[str], doc_index: int) -> float:
        """BM25 score over a pre-tokenized query-term set (the hot path)."""

        tf = self._tfs[doc_index]
        if not tf:
            return 0.0
        denom_len = self._k1 * (1.0 - self._b + self._b * self._doc_len[doc_index] / self._avgdl)
        total = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            total += self._idf.get(term, 0.0) * (freq * (self._k1 + 1.0)) / (freq + denom_len)
        return total


__all__ = ["Bm25Index"]
