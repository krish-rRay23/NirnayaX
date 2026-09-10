"""Shared fixtures for the ML engine and retrieval tests.

Training is deterministic but not free, so the baseline train/eval datasets and a
single fitted model are built once per session and reused across the ML tests.
Tests that need to assert *reproducibility* train their own second model. The
runbook / incident retrievers are likewise built once per session.
"""

from __future__ import annotations

import pytest

from nirnayax.data import build_runbooks, load_dataset
from nirnayax.domain import DatasetSplit
from nirnayax.domain.models import IncidentDataset, Runbook
from nirnayax.ml import TicketDraft, TriageModel, train_triage_model
from nirnayax.retrieval import HybridRetriever, build_incident_retriever, build_runbook_retriever


@pytest.fixture(scope="session")
def full_dataset() -> IncidentDataset:
    return load_dataset("data/all_tickets.csv")


@pytest.fixture(scope="session")
def train_dataset(full_dataset: IncidentDataset) -> IncidentDataset:
    incidents = full_dataset.incidents[:2000]
    from nirnayax.domain.models import DatasetMetadata
    meta = DatasetMetadata(
        name="train_slice",
        split=DatasetSplit.TRAIN,
        seed=20260901,
        size=len(incidents),
        generated_at=full_dataset.metadata.generated_at,
        generator_version=full_dataset.metadata.generator_version,
    )
    return IncidentDataset(metadata=meta, incidents=incidents)


@pytest.fixture(scope="session")
def eval_dataset(full_dataset: IncidentDataset) -> IncidentDataset:
    incidents = full_dataset.incidents[2000:2500]
    from nirnayax.domain.models import DatasetMetadata
    meta = DatasetMetadata(
        name="eval_slice",
        split=DatasetSplit.EVAL,
        seed=20260901,
        size=len(incidents),
        generated_at=full_dataset.metadata.generated_at,
        generator_version=full_dataset.metadata.generator_version,
    )
    return IncidentDataset(metadata=meta, incidents=incidents)


@pytest.fixture(scope="session")
def trained_model(train_dataset: IncidentDataset) -> TriageModel:
    return train_triage_model(train_dataset)


@pytest.fixture(scope="session")
def runbooks() -> tuple[Runbook, ...]:
    return build_runbooks()


@pytest.fixture(scope="session")
def runbook_retriever(runbooks: tuple[Runbook, ...]) -> HybridRetriever:
    return build_runbook_retriever(runbooks)


@pytest.fixture(scope="session")
def incident_retriever(train_dataset: IncidentDataset) -> HybridRetriever:
    return build_incident_retriever(train_dataset.incidents)


@pytest.fixture
def sample_draft() -> TicketDraft:
    """A clear NETWORK / BGP_ROUTING ticket for inference tests."""

    return TicketDraft(
        title="BGP session down between core routers",
        description=(
            "BGP peering session flapping; neighbor state stuck in active, routes "
            "withdrawn and prefixes disappearing from the routing table."
        ),
        affected_service="core-router-edge1",
        region="ap-south-1",
    )
