"""Typed contracts for the retrieval layer: documents, chunks, results, config.

Deliberately free of numpy / scikit-learn imports so the *data contracts* of the
retrieval engine can be constructed, serialized, and tested without the heavy
vector stack. Indexing and scoring live in sibling modules.

The models mirror the domain/ML layers' conventions: immutable (``frozen=True``)
and strict (``extra="forbid"``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: A metadata value: a scalar or a tuple of strings (e.g. tags). ``bool`` is
#: listed first so a boolean is never silently coerced to ``int``.
MetaValue = bool | int | float | str | tuple[str, ...]


class _Frozen(BaseModel):
    """Immutable, strict base (mirrors the domain / ML layers)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Corpus units
# ---------------------------------------------------------------------------
class Document(_Frozen):
    """A source document to be ingested into the retrieval index.

    ``source_type`` / ``source_id`` track provenance back to the originating
    record (a runbook or an incident), which is what citations are built from.
    """

    doc_id: str = Field(min_length=1)
    text: str
    source_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    title: str = ""
    metadata: dict[str, MetaValue] = Field(default_factory=dict)


class Chunk(_Frozen):
    """A retrievable span of a :class:`Document`, with a stable id and offsets.

    ``start`` / ``end`` are character offsets into the parent document's text, so
    a citation can quote the exact passage a result came from.
    """

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    index: int = Field(ge=0)
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    source_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    title: str = ""
    metadata: dict[str, MetaValue] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Results & citations
# ---------------------------------------------------------------------------
class Citation(_Frozen):
    """A traceable pointer from a retrieval result back to its source passage."""

    chunk_id: str
    doc_id: str
    source_type: str
    source_id: str
    title: str
    start: int
    end: int

    def render(self) -> str:
        label = f"{self.source_type} {self.source_id}"
        title = f" {self.title}" if self.title else ""
        cid = self.chunk_id.rsplit("#", 1)[-1]
        return f"[{label} #{cid}]{title} (chars {self.start}-{self.end})"


class RetrievalResult(_Frozen):
    """A ranked chunk with its fused score and per-signal score breakdown."""

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    #: Per-signal scores that fed the ranking, e.g. ``vector`` / ``bm25`` /
    #: ``rrf`` / ``rerank``. Kept open-ended so new signals can be added.
    components: dict[str, float] = Field(default_factory=dict)

    @property
    def citation(self) -> Citation:
        """A :class:`Citation` pointing back to this result's source passage."""

        c = self.chunk
        return Citation(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            source_type=c.source_type,
            source_id=c.source_id,
            title=c.title,
            start=c.start,
            end=c.end,
        )

    def render(self, *, width: int = 96) -> str:
        snippet = " ".join(self.chunk.text.split())
        if len(snippet) > width:
            snippet = snippet[: width - 1].rstrip() + "…"
        return f"#{self.rank} ({self.score:.4f}) {self.citation.render()}\n    {snippet}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ChunkConfig(_Frozen):
    """Character-based chunking settings (sentence-aware, with overlap)."""

    max_chars: int = Field(default=480, ge=1)
    overlap_chars: int = Field(default=80, ge=0)
    #: Trailing fragments shorter than this are merged into the previous chunk.
    min_chunk_chars: int = Field(default=48, ge=0)


class RetrievalConfig(_Frozen):
    """Hybrid retrieval settings shared by all retrievers."""

    top_k: int = Field(default=5, ge=1)
    #: First-stage depth: how many fused candidates enter the rerank stage.
    candidate_k: int = Field(default=20, ge=1)
    #: Reciprocal-rank-fusion constant (larger = flatter contribution by rank).
    rrf_k: int = Field(default=60, ge=1)
    use_reranker: bool = True


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
class RetrievalMetrics(_Frozen):
    """Recall@K, Precision@K, and MRR for a retrieval task."""

    name: str
    n_queries: int = Field(ge=0)
    k_values: tuple[int, ...]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    #: Mean Reciprocal Rank of the first relevant result (1.0 = always rank 1).
    mrr: float

    def render(self) -> str:
        header = f"Retrieval [{self.name}]  n={self.n_queries}  MRR={self.mrr:.4f}"
        ks = sorted(self.k_values)
        recall = "  ".join(f"R@{k}={self.recall_at_k.get(k, 0.0):.4f}" for k in ks)
        prec = "  ".join(f"P@{k}={self.precision_at_k.get(k, 0.0):.4f}" for k in ks)
        lines = [header, "=" * len(header), f"  recall    : {recall}", f"  precision : {prec}"]
        return "\n".join(lines)


__all__ = [
    "Chunk",
    "ChunkConfig",
    "Citation",
    "Document",
    "MetaValue",
    "RetrievalConfig",
    "RetrievalMetrics",
    "RetrievalResult",
]
