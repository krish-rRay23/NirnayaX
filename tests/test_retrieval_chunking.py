"""Tests for document ingestion, chunking, and the record → Document adapters."""

from __future__ import annotations

from nirnayax.domain.models import Runbook
from nirnayax.retrieval.chunking import (
    chunk_document,
    chunk_text,
    document_from_incident,
    document_from_runbook,
    tokenize,
)
from nirnayax.retrieval.types import ChunkConfig, Document


def test_tokenize_lowercases_and_keeps_alnum() -> None:
    assert tokenize("BGP session P99=1840ms!") == ["bgp", "session", "p99", "1840ms"]


def test_tokenize_empty() -> None:
    assert tokenize("   ...  ") == []


def test_chunk_text_empty_returns_nothing() -> None:
    assert chunk_text("") == []


def test_chunk_text_short_is_single_chunk_with_exact_span() -> None:
    text = "A short runbook. It has two sentences."
    spans = chunk_text(text)
    assert len(spans) == 1
    piece, start, end = spans[0]
    assert (start, end) == (0, len(text))
    assert piece == text


def test_chunk_text_span_integrity_and_bounds() -> None:
    # Long, multi-sentence text forces several chunks.
    text = " ".join(f"Sentence number {i} describes an event." for i in range(40))
    config = ChunkConfig(max_chars=120, overlap_chars=30, min_chunk_chars=20)
    spans = chunk_text(text, config)

    assert len(spans) > 1
    for piece, start, end in spans:
        assert text[start:end] == piece  # offsets are exact
    # A single sentence is < max_chars, so packed chunks respect the bound.
    assert all(len(piece) <= config.max_chars for piece, _, _ in spans)


def test_chunk_text_overlap_reuses_trailing_context() -> None:
    text = " ".join(f"Alpha{i} bravo charlie delta." for i in range(30))
    with_overlap = chunk_text(text, ChunkConfig(max_chars=100, overlap_chars=40))
    no_overlap = chunk_text(text, ChunkConfig(max_chars=100, overlap_chars=0))
    # Overlap re-includes trailing segments, so it never yields fewer chunks and
    # consecutive chunks share a region.
    assert len(with_overlap) >= len(no_overlap)
    second_start = with_overlap[1][1]
    first_end = with_overlap[0][2]
    assert second_start < first_end


def test_chunk_document_ids_and_provenance() -> None:
    doc = Document(
        doc_id="D1",
        text=" ".join(f"Step {i} of the procedure runs here." for i in range(20)),
        source_type="runbook",
        source_id="RB-X-001",
        title="Title",
        metadata={"category": "NETWORK"},
    )
    chunks = chunk_document(doc, ChunkConfig(max_chars=90, overlap_chars=10))
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"D1#{i}"
        assert chunk.index == i
        assert chunk.doc_id == "D1"
        assert chunk.source_type == "runbook"
        assert chunk.source_id == "RB-X-001"
        assert chunk.metadata == {"category": "NETWORK"}


def test_document_from_runbook_content_and_metadata(runbooks: tuple[Runbook, ...]) -> None:
    rb = next(r for r in runbooks if r.category.value == "NETWORK")
    doc = document_from_runbook(rb)

    assert doc.source_type == "runbook"
    assert doc.source_id == rb.runbook_id
    assert rb.title in doc.text
    assert rb.summary in doc.text
    # Every step action is present so the whole procedure is retrievable.
    for step in rb.steps:
        assert step.action in doc.text
    assert doc.metadata["category"] == rb.category.value
    assert doc.metadata["subcategory"] == rb.subcategory.value
    assert doc.metadata["tags"] == tuple(rb.tags)


def test_incident_text_has_no_label_leakage() -> None:
    # Build a document straight from a generated incident and assert the
    # ground-truth labels never appear in the retrievable text.
    from nirnayax.data import generate_dataset
    from nirnayax.domain import DatasetSplit

    incident = generate_dataset(5, split=DatasetSplit.EVAL).incidents[0]
    doc = document_from_incident(incident)

    assert incident.title in doc.text
    assert incident.description in doc.text
    assert incident.subcategory.value not in doc.text
    assert incident.category.value not in doc.text
    assert incident.severity.value not in doc.text
    assert incident.priority.value not in doc.text
    # ... but the labels remain available as metadata for filtering.
    assert doc.metadata["subcategory"] == incident.subcategory.value
    assert doc.metadata["priority"] == incident.priority.value


def test_chunk_documents_flattens_multi_docs() -> None:
    from nirnayax.retrieval.chunking import chunk_documents

    d1 = Document(doc_id="D1", text="First doc text.", source_type="t", source_id="s1")
    d2 = Document(doc_id="D2", text="Second doc text.", source_type="t", source_id="s2")
    chunks = chunk_documents([d1, d2])
    assert len(chunks) == 2
    assert chunks[0].doc_id == "D1"
    assert chunks[1].doc_id == "D2"

