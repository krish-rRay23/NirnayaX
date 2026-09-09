"""Document ingestion and chunking.

Turns source records (runbooks, incidents) into :class:`Document` objects and
splits documents into overlapping, character-bounded :class:`Chunk` spans while
tracking exact character offsets so results remain citable.

A single :func:`tokenize` defines the token space shared by the lexical index
(BM25) and the lexical reranker, so they cannot drift apart.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from ..domain.models import Incident, Runbook
from .types import Chunk, ChunkConfig, Document, MetaValue

_TOKEN = re.compile(r"[a-z0-9]+")
#: Sentence / clause boundaries used to find natural split points.
_BOUNDARY = re.compile(r"(?<=[.!?;:\n])\s+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared across lexical scorers."""

    return _TOKEN.findall(text.lower())


def _segments(text: str) -> list[tuple[str, int, int]]:
    """Split text into ``(segment, start, end)`` spans on sentence boundaries.

    Offsets index into the original ``text`` so chunk spans stay accurate.
    """

    spans: list[tuple[str, int, int]] = []
    pos = 0
    for piece in _BOUNDARY.split(text):
        if piece == "":
            continue
        start = text.find(piece, pos)
        if start < 0:  # pragma: no cover - defensive; split pieces are substrings
            start = pos
        end = start + len(piece)
        spans.append((piece, start, end))
        pos = end
    return spans


def chunk_text(text: str, config: ChunkConfig | None = None) -> list[tuple[str, int, int]]:
    """Split ``text`` into ``(chunk, start, end)`` spans, sentence-aware with overlap.

    Segments are greedily packed up to ``max_chars``; consecutive chunks overlap
    by up to ``overlap_chars`` (re-including trailing segments) so context that
    straddles a boundary is not lost. A short trailing chunk is merged back into
    its predecessor. Returned spans use offsets into the original ``text``.
    """

    config = config or ChunkConfig()
    segments = _segments(text)
    if not segments:
        return []

    chunks: list[tuple[int, int]] = []  # (first_seg_index, last_seg_index) inclusive
    i = 0
    n = len(segments)
    while i < n:
        j = i
        length = 0
        while j < n:
            seg_len = segments[j][2] - segments[j][1]
            # Always take at least one segment, even if it alone exceeds max_chars.
            if j > i and length + 1 + seg_len > config.max_chars:
                break
            length += (1 if j > i else 0) + seg_len
            j += 1
        chunks.append((i, j - 1))
        if j >= n:
            break
        # Advance, re-including trailing segments for overlap.
        i = _overlap_start(segments, i, j, config.overlap_chars)

    merged = _merge_short_tail(segments, chunks, config.min_chunk_chars)
    return [(text[s:e], s, e) for s, e in merged]


def _overlap_start(
    segments: Sequence[tuple[str, int, int]], start: int, stop: int, overlap_chars: int
) -> int:
    """Index of the first segment for the next chunk, honoring ``overlap_chars``."""

    if overlap_chars <= 0:
        return stop
    budget = 0
    k = stop - 1
    while k > start:
        seg_len = segments[k][2] - segments[k][1]
        if budget + seg_len > overlap_chars:
            break
        budget += seg_len
        k -= 1
    # Ensure forward progress (never restart at/behind the current chunk's start).
    return max(k + 1, start + 1)


def _merge_short_tail(
    segments: Sequence[tuple[str, int, int]],
    chunks: Sequence[tuple[int, int]],
    min_chunk_chars: int,
) -> list[tuple[int, int]]:
    """Fold a too-short final chunk into the previous one (by char span)."""

    spans = [(segments[a][1], segments[b][2]) for a, b in chunks]
    if len(spans) >= 2 and (spans[-1][1] - spans[-1][0]) < min_chunk_chars:
        prev_start = spans[-2][0]
        last_end = spans[-1][1]
        spans[-2] = (prev_start, last_end)
        spans.pop()
    return spans


def chunk_document(document: Document, config: ChunkConfig | None = None) -> list[Chunk]:
    """Chunk a :class:`Document`, propagating provenance and metadata to each chunk."""

    chunks: list[Chunk] = []
    for index, (piece, start, end) in enumerate(chunk_text(document.text, config)):
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#{index}",
                doc_id=document.doc_id,
                index=index,
                text=piece,
                start=start,
                end=end,
                source_type=document.source_type,
                source_id=document.source_id,
                title=document.title,
                metadata=dict(document.metadata),
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[Document], config: ChunkConfig | None = None
) -> list[Chunk]:
    """Chunk many documents into one flat list (index order preserved)."""

    out: list[Chunk] = []
    for document in documents:
        out.extend(chunk_document(document, config))
    return out


# ---------------------------------------------------------------------------
# Source-record → Document adapters
# ---------------------------------------------------------------------------
def document_from_runbook(runbook: Runbook) -> Document:
    """Render a :class:`Runbook` into a retrievable :class:`Document`.

    The body carries the runbook's natural-language content (title, summary,
    symptoms, ordered steps); the enum labels live only in ``metadata`` so they
    can be used for filtering without polluting the text signal.
    """

    lines = [f"{runbook.title}.", runbook.summary]
    lines.append("Symptoms: " + "; ".join(runbook.symptoms) + ".")
    lines.append("Steps:")
    for step in runbook.steps:
        suffix = f" (expected: {step.expected_signal})" if step.expected_signal else ""
        lines.append(f"{step.order}. {step.action}{suffix}")
    if runbook.tags:
        lines.append("Tags: " + ", ".join(runbook.tags) + ".")
    lines.append(f"Escalation team: {runbook.escalation_team}.")

    metadata: dict[str, MetaValue] = {
        "category": runbook.category.value,
        "subcategory": runbook.subcategory.value,
        "severity_hint": runbook.severity_hint.value,
        "escalation_team": runbook.escalation_team,
        "estimated_resolution_minutes": runbook.estimated_resolution_minutes,
        "tags": tuple(runbook.tags),
    }
    return Document(
        doc_id=runbook.runbook_id,
        text="\n".join(lines),
        source_type="runbook",
        source_id=runbook.runbook_id,
        title=runbook.title,
        metadata=metadata,
    )


def document_from_incident(incident: Incident) -> Document:
    """Render an :class:`Incident` into a retrievable :class:`Document`.

    Only intake-time content (title, description, service/region/channel,
    signals, tags) is placed in the text. The ground-truth ``category`` /
    ``subcategory`` / ``severity`` / ``priority`` labels are kept in ``metadata``
    only — never in the text — so similarity retrieval cannot trivially match on
    the very label an evaluation scores against.
    """

    lines = [f"{incident.title}.", incident.description]
    lines.append(
        f"Service: {incident.affected_service}. Region: {incident.region}. "
        f"Channel: {incident.channel.value}."
    )
    if incident.signals:
        rendered = "; ".join(f"{s.name}={s.value:g}{s.unit}" for s in incident.signals)
        lines.append(f"Signals: {rendered}.")
    if incident.tags:
        lines.append("Tags: " + ", ".join(incident.tags) + ".")

    metadata: dict[str, MetaValue] = {
        "category": incident.category.value,
        "subcategory": incident.subcategory.value,
        "severity": incident.severity.value,
        "priority": incident.priority.value,
        "channel": incident.channel.value,
        "affected_service": incident.affected_service,
        "region": incident.region,
        "customer_impacting": incident.customer_impacting,
        "tags": tuple(incident.tags),
    }
    return Document(
        doc_id=incident.incident_id,
        text="\n".join(lines),
        source_type="incident",
        source_id=incident.incident_id,
        title=incident.title,
        metadata=metadata,
    )


__all__ = [
    "chunk_document",
    "chunk_documents",
    "chunk_text",
    "document_from_incident",
    "document_from_runbook",
    "tokenize",
]
