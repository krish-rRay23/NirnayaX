"""Incident taxonomy and controlled vocabularies for NirnayaX L1 triage.

This module is the single source of truth for the *labels* the triage system
reasons about: the four top-level categories, their subcategories, and the
operational enums (severity, priority, status, channel).

The taxonomy is intentionally **data-driven**: the ``TAXONOMY`` mapping is the
only place that needs to change to add a category or subcategory. Generation,
validation and (future) model layers all derive their behaviour from it, and an
import-time integrity check guarantees the mapping and the ``Subcategory`` enum
never drift apart.
"""

from __future__ import annotations

from enum import StrEnum


class Category(StrEnum):
    """Top-level triage category (the coarse routing label)."""

    NETWORK = "NETWORK"
    APPLICATION_DB = "APPLICATION_DB"
    BILLING_OSS = "BILLING_OSS"
    HARDWARE_ACCESS = "HARDWARE_ACCESS"


class Subcategory(StrEnum):
    """Fine-grained failure mode. Each value maps to exactly one ``Category``."""

    # --- NETWORK ---
    LATENCY_PACKET_LOSS = "LATENCY_PACKET_LOSS"
    LINK_DOWN = "LINK_DOWN"
    DNS_RESOLUTION = "DNS_RESOLUTION"
    BGP_ROUTING = "BGP_ROUTING"

    # --- APPLICATION_DB ---
    CONNECTION_POOL_EXHAUSTION = "CONNECTION_POOL_EXHAUSTION"
    REPLICATION_LAG = "REPLICATION_LAG"
    DEADLOCK = "DEADLOCK"
    SLOW_QUERY = "SLOW_QUERY"
    DISK_SPACE = "DISK_SPACE"

    # --- BILLING_OSS ---
    RATING_ENGINE_ERROR = "RATING_ENGINE_ERROR"
    INVOICE_GENERATION_FAILURE = "INVOICE_GENERATION_FAILURE"
    MEDIATION_FEED_GAP = "MEDIATION_FEED_GAP"
    PROVISIONING_SYNC_FAILURE = "PROVISIONING_SYNC_FAILURE"

    # --- HARDWARE_ACCESS ---
    ACCOUNT_LOCKOUT = "ACCOUNT_LOCKOUT"
    VPN_ACCESS_FAILURE = "VPN_ACCESS_FAILURE"
    SERVER_HARDWARE_FAULT = "SERVER_HARDWARE_FAULT"
    PERIPHERAL_FAILURE = "PERIPHERAL_FAILURE"


class Severity(StrEnum):
    """Technical impact of the incident (SEV1 = most severe)."""

    SEV1 = "SEV1"
    SEV2 = "SEV2"
    SEV3 = "SEV3"
    SEV4 = "SEV4"


class Priority(StrEnum):
    """Work-queue priority (P1 = handle first)."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class IncidentStatus(StrEnum):
    """Lifecycle state of an incident ticket."""

    NEW = "NEW"
    TRIAGED = "TRIAGED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class Channel(StrEnum):
    """How the incident entered the queue."""

    MONITORING = "MONITORING"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CHAT = "CHAT"
    SELF_SERVICE = "SELF_SERVICE"


# ---------------------------------------------------------------------------
# The taxonomy: Category -> ordered subcategories.
# ---------------------------------------------------------------------------
TAXONOMY: dict[Category, tuple[Subcategory, ...]] = {
    Category.NETWORK: (
        Subcategory.LATENCY_PACKET_LOSS,
        Subcategory.LINK_DOWN,
        Subcategory.DNS_RESOLUTION,
        Subcategory.BGP_ROUTING,
    ),
    Category.APPLICATION_DB: (
        Subcategory.CONNECTION_POOL_EXHAUSTION,
        Subcategory.REPLICATION_LAG,
        Subcategory.DEADLOCK,
        Subcategory.SLOW_QUERY,
        Subcategory.DISK_SPACE,
    ),
    Category.BILLING_OSS: (
        Subcategory.RATING_ENGINE_ERROR,
        Subcategory.INVOICE_GENERATION_FAILURE,
        Subcategory.MEDIATION_FEED_GAP,
        Subcategory.PROVISIONING_SYNC_FAILURE,
    ),
    Category.HARDWARE_ACCESS: (
        Subcategory.ACCOUNT_LOCKOUT,
        Subcategory.VPN_ACCESS_FAILURE,
        Subcategory.SERVER_HARDWARE_FAULT,
        Subcategory.PERIPHERAL_FAILURE,
    ),
}

_SUBCATEGORY_TO_CATEGORY: dict[Subcategory, Category] = {
    sub: cat for cat, subs in TAXONOMY.items() for sub in subs
}


def category_of(subcategory: Subcategory) -> Category:
    """Return the parent :class:`Category` for ``subcategory``."""

    return _SUBCATEGORY_TO_CATEGORY[subcategory]


def subcategories_of(category: Category) -> tuple[Subcategory, ...]:
    """Return the ordered subcategories belonging to ``category``."""

    return TAXONOMY[category]


def all_subcategories() -> tuple[Subcategory, ...]:
    """Return every subcategory, grouped by category order."""

    return tuple(_SUBCATEGORY_TO_CATEGORY)


def _validate_taxonomy() -> None:
    """Fail fast at import time if the enum and mapping disagree."""

    mapped = [sub for subs in TAXONOMY.values() for sub in subs]
    mapped_set = set(mapped)

    if len(mapped) != len(mapped_set):
        raise RuntimeError("Duplicate subcategory detected in TAXONOMY mapping.")

    missing = set(Subcategory) - mapped_set
    if missing:
        names = sorted(sub.value for sub in missing)
        raise RuntimeError(f"Subcategories missing from TAXONOMY: {names}")


_validate_taxonomy()



