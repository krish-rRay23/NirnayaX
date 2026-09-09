"""Phase 3 retrieval engine: hybrid RAG + incident correlation.

A provider-agnostic retrieval layer over the Phase-1 knowledge base (runbooks)
and historical incidents. It combines dense vector search with lexical BM25,
fuses the two with Reciprocal Rank Fusion, optionally reranks, supports metadata
filtering, and returns cited results. Embedding and reranking are defined by
:class:`Embedder` / :class:`Reranker` protocols so a future agent can plug in an
LLM / cross-encoder without touching the retrieval logic; the defaults here
(:class:`TfidfEmbedder`, :class:`LexicalReranker`) are deterministic and offline.

Requires the optional ``retrieval`` extra (``pip install -e ".[retrieval]"``),
which pulls in numpy + scikit-learn. No LLM, agent, or ticketing code lives here.

Typical use::

    from nirnayax.data import build_runbooks
    from nirnayax.retrieval import build_runbook_retriever

    retriever = build_runbook_retriever(build_runbooks())
    for hit in retriever.retrieve("BGP session keeps flapping", k=3):
        print(hit.render())
"""

from __future__ import annotations

from .chunking import (
    chunk_document,
    chunk_documents,
    chunk_text,
    document_from_incident,
    document_from_runbook,
    tokenize,
)
from .embeddings import Embedder, EmbeddingMatrix, TfidfEmbedder, cosine_similarity
from .evaluation import (
    evaluate_incident_retrieval,
    evaluate_runbook_retrieval,
    precision_at_k,
    ranked_source_ids,
    recall_at_k,
    reciprocal_rank,
)
from .lexical import Bm25Index
from .ranking import LexicalReranker, Reranker, reciprocal_rank_fusion
from .retriever import (
    Filters,
    HybridRetriever,
    build_incident_retriever,
    build_runbook_retriever,
    match_metadata,
    similar_incidents,
)
from .types import (
    Chunk,
    ChunkConfig,
    Citation,
    Document,
    MetaValue,
    RetrievalConfig,
    RetrievalMetrics,
    RetrievalResult,
)
from .vector import VectorIndex

__all__ = [
    "Bm25Index",
    "Chunk",
    "ChunkConfig",
    "Citation",
    "Document",
    "Embedder",
    "EmbeddingMatrix",
    "Filters",
    "HybridRetriever",
    "LexicalReranker",
    "MetaValue",
    "Reranker",
    "RetrievalConfig",
    "RetrievalMetrics",
    "RetrievalResult",
    "TfidfEmbedder",
    "VectorIndex",
    "build_incident_retriever",
    "build_runbook_retriever",
    "chunk_document",
    "chunk_documents",
    "chunk_text",
    "cosine_similarity",
    "document_from_incident",
    "document_from_runbook",
    "evaluate_incident_retrieval",
    "evaluate_runbook_retrieval",
    "match_metadata",
    "precision_at_k",
    "ranked_source_ids",
    "recall_at_k",
    "reciprocal_rank",
    "reciprocal_rank_fusion",
    "similar_incidents",
    "tokenize",
]
