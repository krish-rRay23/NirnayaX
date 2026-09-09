"""Feature extraction: turn an incident or an incoming ticket into model text.

A single :func:`build_text` is the *only* place raw fields become model input, so
training and inference cannot drift apart. Only fields available at intake time
are used — deliberately **excluding** ``severity``, ``priority``,
``customer_impacting`` and ``status`` to avoid label leakage (priority in
particular is derived from severity + customer impact).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..domain.models import Incident
from ..domain.taxonomy import Channel
from .types import FeatureConfig

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class TicketDraft(BaseModel):
    """An incoming, unlabelled ticket — the inference input.

    Only ``title`` and ``description`` are required; the optional fields mirror
    what an intake form or monitoring alert typically carries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=2000)
    affected_service: str | None = Field(default=None, max_length=80)
    region: str | None = Field(default=None, max_length=40)
    channel: Channel | None = None
    tags: tuple[str, ...] = ()


def _norm(text: str) -> str:
    """Collapse to a single lowercase token (for categorical fields)."""

    return _NON_ALNUM.sub("_", text.lower()).strip("_")


def build_text(
    *,
    title: str,
    description: str,
    affected_service: str | None,
    region: str | None,
    channel: str | None,
    tags: Sequence[str],
    config: FeatureConfig,
) -> str:
    """Compose the single text string fed to the vectorizer.

    Prose (title + description) is kept verbatim; categorical fields are emitted
    as namespaced tokens (``svc_…``, ``region_…``, ``chan_…``, ``tag_…``) so the
    vectorizer treats them as discrete features rather than free text.
    """

    parts = [title, description]
    if config.include_metadata:
        if affected_service:
            parts.append(f"svc_{_norm(affected_service)}")
        if region:
            parts.append(f"region_{_norm(region)}")
        if channel:
            parts.append(f"chan_{_norm(channel)}")
        parts.extend(f"tag_{_norm(tag)}" for tag in tags if tag)
    return " ".join(parts)


def text_from_incident(incident: Incident, config: FeatureConfig) -> str:
    """Featurize a labelled :class:`Incident` (training / evaluation)."""

    return build_text(
        title=incident.title,
        description=incident.description,
        affected_service=incident.affected_service,
        region=incident.region,
        channel=incident.channel.value,
        tags=incident.tags,
        config=config,
    )


def text_from_draft(draft: TicketDraft, config: FeatureConfig) -> str:
    """Featurize an incoming :class:`TicketDraft` (inference)."""

    return build_text(
        title=draft.title,
        description=draft.description,
        affected_service=draft.affected_service,
        region=draft.region,
        channel=draft.channel.value if draft.channel is not None else None,
        tags=draft.tags,
        config=config,
    )


__all__ = ["TicketDraft", "build_text", "text_from_draft", "text_from_incident"]
