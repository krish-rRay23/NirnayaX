"""Unit tests for feature extraction and the inference input contract.

The central guarantee here is **no label leakage**: the featurizer must use only
fields available at intake time and never ``severity``, ``priority``,
``customer_impacting`` or ``status`` (priority is derived from the first two).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nirnayax.data import generate_incidents
from nirnayax.domain.taxonomy import Channel, IncidentStatus, Priority, Severity
from nirnayax.ml import FeatureConfig, TicketDraft, text_from_draft, text_from_incident
from nirnayax.ml.features import build_text


def test_prose_is_kept_verbatim() -> None:
    text = build_text(
        title="Latency spike",
        description="p99 climbed to 1840ms",
        affected_service=None,
        region=None,
        channel=None,
        tags=(),
        config=FeatureConfig(),
    )
    assert "Latency spike" in text
    assert "p99 climbed to 1840ms" in text


def test_metadata_becomes_namespaced_tokens() -> None:
    text = build_text(
        title="t",
        description="d",
        affected_service="Core Router",
        region="AP-South-1",
        channel="MONITORING",
        tags=("noc", "urgent"),
        config=FeatureConfig(include_metadata=True),
    )
    assert "svc_core_router" in text
    assert "region_ap_south_1" in text
    assert "chan_monitoring" in text
    assert "tag_noc" in text
    assert "tag_urgent" in text


def test_metadata_can_be_disabled() -> None:
    config = FeatureConfig(include_metadata=False)
    text = build_text(
        title="t",
        description="d",
        affected_service="core-router",
        region="ap-south-1",
        channel="MONITORING",
        tags=("noc",),
        config=config,
    )
    assert text == "t d"


def test_incident_and_equivalent_draft_agree() -> None:
    """text_from_incident and text_from_draft must not drift for the same fields."""

    inc = generate_incidents(1, seed=5)[0]
    draft = TicketDraft(
        title=inc.title,
        description=inc.description,
        affected_service=inc.affected_service,
        region=inc.region,
        channel=inc.channel,
        tags=inc.tags,
    )
    config = FeatureConfig()
    assert text_from_incident(inc, config) == text_from_draft(draft, config)


def test_features_ignore_label_derived_fields() -> None:
    """Changing severity/priority/customer_impacting/status must not change the text."""

    inc = generate_incidents(1, seed=9)[0]
    flipped = inc.model_copy(
        update={
            "severity": Severity.SEV1 if inc.severity is not Severity.SEV1 else Severity.SEV4,
            "priority": Priority.P1 if inc.priority is not Priority.P1 else Priority.P4,
            "customer_impacting": not inc.customer_impacting,
            "status": IncidentStatus.CLOSED,
        }
    )
    config = FeatureConfig()
    assert text_from_incident(inc, config) == text_from_incident(flipped, config)


def test_ticket_draft_rejects_leakage_fields() -> None:
    """The inference contract must not even accept label-derived fields."""

    for leak in ("severity", "priority", "customer_impacting", "status"):
        with pytest.raises(ValidationError):
            TicketDraft.model_validate({"title": "x", "description": "ten chars!", leak: "SEV1"})


def test_ticket_draft_requires_title_and_description() -> None:
    with pytest.raises(ValidationError):
        TicketDraft(title="", description="ten chars!")
    with pytest.raises(ValidationError):
        TicketDraft(title="ok", description="")


def test_ticket_draft_is_frozen() -> None:
    draft = TicketDraft(title="t", description="ten chars!")
    with pytest.raises(ValidationError):
        draft.title = "changed"  # type: ignore[misc]


def test_channel_enum_is_accepted() -> None:
    draft = TicketDraft(title="t", description="ten chars!", channel=Channel.EMAIL)
    text = text_from_draft(draft, FeatureConfig())
    assert "chan_email" in text
