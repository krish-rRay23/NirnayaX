"""Tests for the TF-IDF embedder and cosine helper."""

from __future__ import annotations

import numpy as np
import pytest

from nirnayax.retrieval.embeddings import Embedder, TfidfEmbedder, cosine_similarity

_CORPUS = [
    "bgp session flapping between core routers",
    "database connection pool exhausted clients timeout",
    "dns resolution failing nxdomain across region",
]


def test_fit_and_embed_shapes() -> None:
    embedder = TfidfEmbedder.fit(_CORPUS)
    matrix = embedder.embed(_CORPUS)
    assert matrix.shape == (3, embedder.dim)
    assert embedder.dim > 0
    assert embedder.name == "tfidf"


def test_rows_are_l2_normalized() -> None:
    embedder = TfidfEmbedder.fit(_CORPUS)
    norms = np.linalg.norm(embedder.embed(_CORPUS), axis=1)
    assert np.allclose(norms, 1.0)


def test_cosine_identical_is_one_disjoint_is_zero() -> None:
    embedder = TfidfEmbedder.fit(_CORPUS)
    matrix = embedder.embed(_CORPUS)
    same = embedder.embed(["bgp session flapping between core routers"])[0]
    scores = cosine_similarity(matrix, same)
    assert scores[0] == pytest.approx(1.0, abs=1e-9)  # identical to doc 0
    assert scores[1] == pytest.approx(0.0, abs=1e-9)  # disjoint vocabulary


def test_unknown_only_query_is_zero_vector() -> None:
    embedder = TfidfEmbedder.fit(_CORPUS)
    vec = embedder.embed(["kubernetes helm istio"])[0]
    assert float(np.linalg.norm(vec)) == pytest.approx(0.0)


def test_embedding_is_deterministic() -> None:
    a = TfidfEmbedder.fit(_CORPUS).embed(_CORPUS)
    b = TfidfEmbedder.fit(_CORPUS).embed(_CORPUS)
    assert np.array_equal(a, b)


def test_fit_empty_corpus_raises() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        TfidfEmbedder.fit([])


def test_satisfies_embedder_protocol() -> None:
    assert isinstance(TfidfEmbedder.fit(_CORPUS), Embedder)
