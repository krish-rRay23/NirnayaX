"""Synthetic troubleshooting runbook catalog.

:func:`build_runbooks` deterministically constructs one :class:`Runbook` per
subcategory from a compact spec table, so every triage label has an associated
remediation guide. Runbook ids are stable (``RB-<CATEGORY>-<NNN>``, numbered in
taxonomy order) and cross-references between related runbooks are resolved to
those ids. This catalog is the (future) retrieval knowledge base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..domain.models import Runbook, RunbookStep
from ..domain.taxonomy import Category, Severity, Subcategory, category_of

#: Review date stamped on every generated runbook (keeps output reproducible).
LAST_REVIEWED = date(2026, 7, 15)

_Step = tuple[str, str | None]


@dataclass(frozen=True, slots=True)
class _RunbookSpec:
    subcategory: Subcategory
    title: str
    summary: str
    symptoms: tuple[str, ...]
    steps: tuple[_Step, ...]
    escalation_team: str
    severity_hint: Severity
    estimated_resolution_minutes: int
    tags: tuple[str, ...]
    related: tuple[Subcategory, ...] = field(default_factory=tuple)


_RUNBOOK_SPECS: tuple[_RunbookSpec, ...] = (
    # ------------------------------ NETWORK ------------------------------
    _RunbookSpec(
        subcategory=Subcategory.LATENCY_PACKET_LOSS,
        title="Diagnose elevated latency and packet loss",
        summary="Triage and mitigate packet loss or latency on a network element affecting traffic.",
        symptoms=(
            "Monitoring alerts for packet loss above threshold",
            "Increased RTT/jitter on links",
            "Customer reports of slow or dropping connections",
        ),
        steps=(
            (
                "Confirm scope: identify the affected element, region and interfaces from the alert and NMS dashboards.",
                "Impacted device/interfaces identified",
            ),
            (
                "Check interface error/discard counters and utilization on the affected links.",
                "Errored or saturated interface located",
            ),
            (
                "Run path diagnostics (ping/traceroute/MTR) from probes to isolate the lossy hop.",
                "Lossy hop or congested segment identified",
            ),
            (
                "If a single link is faulty, shift traffic to a redundant path or drain the link.",
                "Loss returns to baseline after reroute",
            ),
            (
                "If congestion is the cause, apply QoS/traffic-engineering changes or raise a capacity request.",
                None,
            ),
            ("Escalate to network engineering if loss persists after mitigation.", None),
        ),
        escalation_team="NOC-Network",
        severity_hint=Severity.SEV2,
        estimated_resolution_minutes=45,
        tags=("network", "latency", "packet-loss"),
        related=(Subcategory.LINK_DOWN, Subcategory.BGP_ROUTING),
    ),
    _RunbookSpec(
        subcategory=Subcategory.LINK_DOWN,
        title="Restore a downed link or circuit",
        summary="Respond to one or more network interfaces or circuits reporting loss of carrier.",
        symptoms=(
            "Interface DOWN / loss-of-carrier alarms",
            "Loss of reachability to a site or segment",
            "Redundant path carrying full load",
        ),
        steps=(
            (
                "Identify the down interfaces and whether redundancy is still protecting traffic.",
                "Redundancy status known",
            ),
            (
                "Verify physical layer: SFP/optic light levels, cabling and patch status.",
                "Physical fault confirmed or excluded",
            ),
            (
                "Check for planned maintenance or carrier outages on the affected circuit.",
                "Maintenance/carrier status confirmed",
            ),
            (
                "Attempt interface reset/bounce if safe; confirm the link re-establishes.",
                "Link returns UP",
            ),
            (
                "If a carrier circuit is down, raise a ticket with the provider and dispatch field engineering.",
                None,
            ),
        ),
        escalation_team="NOC-Network",
        severity_hint=Severity.SEV1,
        estimated_resolution_minutes=90,
        tags=("network", "link", "outage"),
        related=(Subcategory.LATENCY_PACKET_LOSS,),
    ),
    _RunbookSpec(
        subcategory=Subcategory.DNS_RESOLUTION,
        title="Resolve DNS resolution failures",
        summary="Diagnose SERVFAIL or slow DNS responses impacting service discovery.",
        symptoms=(
            "Elevated SERVFAIL / NXDOMAIN rates",
            "High query latency",
            "Downstream services unable to resolve endpoints",
        ),
        steps=(
            (
                "Confirm which resolver(s) and zones are failing from monitoring and query logs.",
                "Failing resolver/zone identified",
            ),
            (
                "Test resolution directly (dig/nslookup) against the affected and a healthy resolver.",
                "Failure reproduced",
            ),
            (
                "Check resolver health: CPU, cache size, upstream forwarders and recent config/zone changes.",
                "Root cause candidate found",
            ),
            (
                "Fail over to a healthy resolver or roll back the offending change.",
                "Resolution success rate recovers",
            ),
            ("Flush poisoned cache entries if stale/incorrect records are served.", None),
        ),
        escalation_team="NOC-Network",
        severity_hint=Severity.SEV2,
        estimated_resolution_minutes=40,
        tags=("network", "dns"),
    ),
    _RunbookSpec(
        subcategory=Subcategory.BGP_ROUTING,
        title="Stabilize flapping BGP sessions",
        summary="Address BGP session instability and route withdrawals affecting reachability.",
        symptoms=(
            "BGP neighbor state flapping",
            "Large numbers of routes withdrawn/announced",
            "Intermittent reachability to peered networks",
        ),
        steps=(
            (
                "Identify the flapping neighbors and the peer/transit they connect to.",
                "Affected sessions identified",
            ),
            (
                "Correlate with link errors or latency on the underlying interfaces.",
                "Underlying transport checked",
            ),
            (
                "Review recent policy/prefix-list/route-map changes on the affected routers.",
                "Recent changes reviewed",
            ),
            (
                "Apply session dampening or administratively reset the unstable session in a controlled window.",
                "Session stabilizes",
            ),
            ("Coordinate with the peer's NOC if the instability originates upstream.", None),
        ),
        escalation_team="NOC-Network",
        severity_hint=Severity.SEV2,
        estimated_resolution_minutes=60,
        tags=("network", "bgp", "routing"),
        related=(Subcategory.LATENCY_PACKET_LOSS,),
    ),
    # --------------------------- APPLICATION_DB --------------------------
    _RunbookSpec(
        subcategory=Subcategory.CONNECTION_POOL_EXHAUSTION,
        title="Relieve database connection pool exhaustion",
        summary="Restore service when an application's DB connection pool is saturated.",
        symptoms=(
            "Pool-timeout / 'cannot get connection' errors",
            "High connection-checkout wait times",
            "Rising request error rate with healthy DB CPU",
        ),
        steps=(
            (
                "Confirm pool utilization and checkout wait times for the affected service.",
                "Pool saturation confirmed",
            ),
            (
                "Identify connection leaks or long-running transactions holding connections.",
                "Leaking/long queries identified",
            ),
            (
                "Check the database for the number of active vs. idle-in-transaction sessions.",
                "DB session mix reviewed",
            ),
            (
                "Terminate stuck sessions and, if safe, restart the offending app instances to reset the pool.",
                "Pool utilization drops",
            ),
            (
                "Tune pool size / timeouts or scale out the service; file a follow-up for the leak.",
                None,
            ),
        ),
        escalation_team="DBA-OnCall",
        severity_hint=Severity.SEV2,
        estimated_resolution_minutes=40,
        tags=("application", "database", "connection-pool"),
        related=(Subcategory.SLOW_QUERY, Subcategory.DEADLOCK),
    ),
    _RunbookSpec(
        subcategory=Subcategory.REPLICATION_LAG,
        title="Reduce database replication lag",
        summary="Bring a lagging read replica back within acceptable bounds.",
        symptoms=(
            "Replica lag above SLO",
            "Growing WAL/redo backlog",
            "Stale reads reported by consumers",
        ),
        steps=(
            (
                "Confirm the lag magnitude and which replica(s) are affected.",
                "Lagging replicas identified",
            ),
            (
                "Check replica resource saturation (CPU, disk I/O) and network to the primary.",
                "Bottleneck located",
            ),
            (
                "Look for long-running queries or locks on the replica blocking apply.",
                "Blocking activity found",
            ),
            (
                "Pause heavy read workloads or add apply resources until the backlog drains.",
                "Lag trends downward",
            ),
            ("If lag is unrecoverable, rebuild/reseed the replica from a fresh snapshot.", None),
        ),
        escalation_team="DBA-OnCall",
        severity_hint=Severity.SEV3,
        estimated_resolution_minutes=60,
        tags=("application", "database", "replication"),
        related=(Subcategory.DISK_SPACE,),
    ),
    _RunbookSpec(
        subcategory=Subcategory.DEADLOCK,
        title="Break a database deadlock storm",
        summary="Reduce deadlocks and aborted transactions degrading an application.",
        symptoms=(
            "Deadlock-victim errors in application logs",
            "Elevated transaction rollback rate",
            "Long lock-wait times",
        ),
        steps=(
            (
                "Confirm the deadlock rate and capture deadlock graphs from the DB.",
                "Deadlock graph captured",
            ),
            (
                "Identify the conflicting statements and the tables/indexes involved.",
                "Conflicting queries identified",
            ),
            (
                "Terminate the highest-impact blocking transactions to clear the immediate storm.",
                "Deadlock rate drops",
            ),
            ("Advise the app team to align lock ordering / reduce transaction scope.", None),
            ("Add or adjust indexes to shorten locks; schedule a code fix follow-up.", None),
        ),
        escalation_team="DBA-OnCall",
        severity_hint=Severity.SEV2,
        estimated_resolution_minutes=45,
        tags=("application", "database", "deadlock", "concurrency"),
        related=(Subcategory.SLOW_QUERY, Subcategory.CONNECTION_POOL_EXHAUSTION),
    ),
    _RunbookSpec(
        subcategory=Subcategory.SLOW_QUERY,
        title="Investigate slow database queries",
        summary="Restore query performance when p99 latency breaches SLO.",
        symptoms=(
            "p99 query latency above SLO",
            "High share of slow queries",
            "Elevated DB CPU or I/O",
        ),
        steps=(
            (
                "Identify the top slow statements from the slow-query log / performance views.",
                "Offending queries identified",
            ),
            (
                "Capture and review execution plans for plan regressions or missing indexes.",
                "Plan issue confirmed",
            ),
            ("Check for stale statistics and refresh them where appropriate.", None),
            (
                "Add/adjust indexes or apply query hints; consider caching hot reads.",
                "p99 latency recovers",
            ),
            ("If a recent deploy caused the regression, coordinate a rollback.", None),
        ),
        escalation_team="DBA-OnCall",
        severity_hint=Severity.SEV3,
        estimated_resolution_minutes=50,
        tags=("application", "database", "slow-query", "performance"),
        related=(Subcategory.DEADLOCK, Subcategory.CONNECTION_POOL_EXHAUSTION),
    ),
    _RunbookSpec(
        subcategory=Subcategory.DISK_SPACE,
        title="Recover low database disk space",
        summary="Prevent write failures when a data volume approaches capacity.",
        symptoms=(
            "Disk usage above critical threshold",
            "Low free space / inode exhaustion",
            "Imminent or occurring write failures",
        ),
        steps=(
            (
                "Confirm the affected volume and current used/free space and inode usage.",
                "Capacity state confirmed",
            ),
            (
                "Identify the largest consumers: WAL/redo, logs, temp, bloated tables.",
                "Top space consumers identified",
            ),
            (
                "Reclaim space safely: rotate/ship logs, archive WAL, purge temp files.",
                "Free space recovered",
            ),
            (
                "Extend the volume or add storage if reclamation is insufficient.",
                "Usage below threshold",
            ),
            ("Schedule vacuum/compaction and review retention to prevent recurrence.", None),
        ),
        escalation_team="DBA-OnCall",
        severity_hint=Severity.SEV2,
        estimated_resolution_minutes=35,
        tags=("application", "database", "disk", "capacity"),
        related=(Subcategory.REPLICATION_LAG,),
    ),
    # ---------------------------- BILLING_OSS ----------------------------
    _RunbookSpec(
        subcategory=Subcategory.RATING_ENGINE_ERROR,
        title="Recover rating engine errors",
        summary="Restore CDR rating and prevent revenue leakage when rating fails.",
        symptoms=(
            "Rising rated-record error rate",
            "CDRs stuck in an error queue",
            "Revenue assurance alerts",
        ),
        steps=(
            (
                "Quantify the failure: number of errored CDRs and the affected tariff/plan.",
                "Failure scope quantified",
            ),
            (
                "Inspect rating logs for the dominant error (tariff lookup, config, dependency).",
                "Dominant error identified",
            ),
            (
                "Validate recent tariff/config changes and roll back if they introduced the fault.",
                None,
            ),
            ("Reprocess the errored CDR batch once the fault is corrected.", "Error queue drains"),
            ("Reconcile with revenue assurance and file a follow-up for permanent fix.", None),
        ),
        escalation_team="Billing-Engineering",
        severity_hint=Severity.SEV2,
        estimated_resolution_minutes=75,
        tags=("billing", "oss", "rating", "revenue"),
        related=(Subcategory.MEDIATION_FEED_GAP,),
    ),
    _RunbookSpec(
        subcategory=Subcategory.INVOICE_GENERATION_FAILURE,
        title="Recover a failed invoicing batch",
        summary="Complete invoice generation when the billing batch fails or stalls.",
        symptoms=(
            "Invoicing batch failed or delayed",
            "Accounts missing invoices for the cycle",
            "Bill-cycle SLA at risk",
        ),
        steps=(
            (
                "Confirm how many accounts failed and the batch's current state/checkpoint.",
                "Failure count and state known",
            ),
            (
                "Review batch logs for the failing stage (data, template, downstream service).",
                "Failing stage identified",
            ),
            ("Fix or bypass the blocking dependency (e.g., document service, storage).", None),
            (
                "Re-run the batch from the last good checkpoint for the failed accounts only.",
                "Invoices generated",
            ),
            ("Verify totals against expected volumes and notify billing operations.", None),
        ),
        escalation_team="Billing-Engineering",
        severity_hint=Severity.SEV3,
        estimated_resolution_minutes=90,
        tags=("billing", "oss", "invoicing", "batch"),
        related=(Subcategory.RATING_ENGINE_ERROR,),
    ),
    _RunbookSpec(
        subcategory=Subcategory.MEDIATION_FEED_GAP,
        title="Close a mediation feed gap",
        summary="Recover missing usage feeds so downstream rating is not starved.",
        symptoms=(
            "Expected mediation files missing",
            "Feed delayed beyond schedule",
            "Downstream rating volumes dropping",
        ),
        steps=(
            (
                "Identify which feeds/files are missing and the source network elements.",
                "Missing feeds identified",
            ),
            (
                "Check collector connectivity and credentials to the source systems.",
                "Connectivity verified",
            ),
            (
                "Re-pull or re-request the missing files from the source or its archive.",
                "Files retrieved",
            ),
            (
                "Re-ingest the recovered files and confirm record counts reconcile.",
                "Counts reconcile",
            ),
            ("Backfill downstream rating for the recovered window.", None),
        ),
        escalation_team="Billing-Engineering",
        severity_hint=Severity.SEV3,
        estimated_resolution_minutes=70,
        tags=("billing", "oss", "mediation", "usage"),
        related=(Subcategory.RATING_ENGINE_ERROR,),
    ),
    _RunbookSpec(
        subcategory=Subcategory.PROVISIONING_SYNC_FAILURE,
        title="Clear stuck provisioning synchronization",
        summary="Unblock orders that fail to synchronize activations to the network.",
        symptoms=(
            "Orders stuck in a pending/failed state",
            "Retries exhausted against the network",
            "New activations not completing",
        ),
        steps=(
            (
                "Quantify stuck orders and the step at which synchronization fails.",
                "Stuck step identified",
            ),
            (
                "Check connectivity/health of the target network provisioning systems.",
                "Downstream health verified",
            ),
            (
                "Inspect a sample failed order for the specific downstream error.",
                "Root error identified",
            ),
            (
                "Fix the dependency and replay the stuck orders in controlled batches.",
                "Orders complete",
            ),
            ("Confirm activations on the network and reconcile order status.", None),
        ),
        escalation_team="Billing-Engineering",
        severity_hint=Severity.SEV2,
        estimated_resolution_minutes=80,
        tags=("billing", "oss", "provisioning", "activation"),
    ),
    # -------------------------- HARDWARE_ACCESS --------------------------
    _RunbookSpec(
        subcategory=Subcategory.ACCOUNT_LOCKOUT,
        title="Resolve user account lockouts",
        summary="Restore access for users locked out and rule out malicious activity.",
        symptoms=(
            "Multiple users reporting lockouts",
            "Spike in failed login / lockout events",
            "Authentication denied at the IdP",
        ),
        steps=(
            (
                "Confirm the number and scope of locked accounts from the IdP/directory.",
                "Lockout scope confirmed",
            ),
            (
                "Check for a lockout-policy change, expired credentials, or a sync issue.",
                "Cause candidate found",
            ),
            (
                "Rule out brute-force/credential-stuffing by reviewing source IPs and patterns.",
                "Malicious activity assessed",
            ),
            (
                "Unlock verified users and reset credentials where required.",
                "Users can authenticate",
            ),
            ("If an attack is suspected, engage security and enforce additional controls.", None),
        ),
        escalation_team="IT-Helpdesk",
        severity_hint=Severity.SEV3,
        estimated_resolution_minutes=30,
        tags=("access", "identity", "lockout"),
        related=(Subcategory.VPN_ACCESS_FAILURE,),
    ),
    _RunbookSpec(
        subcategory=Subcategory.VPN_ACCESS_FAILURE,
        title="Restore VPN access",
        summary="Recover remote-access connectivity when users cannot establish tunnels.",
        symptoms=(
            "Users unable to connect to VPN",
            "Tunnel-setup / authentication failures",
            "Remote workforce impacted",
        ),
        steps=(
            (
                "Confirm scope: which gateway/region and how many users are affected.",
                "Affected gateway identified",
            ),
            (
                "Check gateway health, capacity/licensing and certificate validity.",
                "Gateway health verified",
            ),
            (
                "Verify authentication backend (IdP/RADIUS) reachability and recent changes.",
                "Auth path verified",
            ),
            (
                "Fail over to a healthy gateway or restore capacity/certificates.",
                "Connections succeed",
            ),
            ("Communicate status to users and monitor reconnection rates.", None),
        ),
        escalation_team="IT-Helpdesk",
        severity_hint=Severity.SEV2,
        estimated_resolution_minutes=40,
        tags=("access", "vpn", "remote"),
        related=(Subcategory.ACCOUNT_LOCKOUT,),
    ),
    _RunbookSpec(
        subcategory=Subcategory.SERVER_HARDWARE_FAULT,
        title="Respond to a server hardware fault",
        summary="Handle component failures or thermal alarms on physical servers.",
        symptoms=(
            "Hardware alarm from BMC/iLO/iDRAC",
            "Failed component (disk, PSU, fan, memory)",
            "Elevated chassis temperature",
        ),
        steps=(
            (
                "Identify the host, failed component(s) and current redundancy state.",
                "Failed component identified",
            ),
            (
                "Confirm workloads are protected: migrate/evacuate VMs or drain the node.",
                "Workloads protected",
            ),
            (
                "Check environmental factors (cooling, airflow) for thermal alarms.",
                "Environment verified",
            ),
            (
                "Raise a hardware replacement request and dispatch datacenter operations.",
                "Replacement scheduled",
            ),
            ("After replacement, validate health and return the node to service.", None),
        ),
        escalation_team="DC-Operations",
        severity_hint=Severity.SEV2,
        estimated_resolution_minutes=120,
        tags=("hardware", "server", "fault"),
    ),
    _RunbookSpec(
        subcategory=Subcategory.PERIPHERAL_FAILURE,
        title="Resolve end-user peripheral failures",
        summary="Restore end-user devices/peripherals reported non-functional at a site.",
        symptoms=(
            "Devices non-functional at a site",
            "Users unable to complete workflow",
            "Suspected hardware or driver fault",
        ),
        steps=(
            (
                "Confirm the affected devices, site and common factor (model, driver, network).",
                "Common factor identified",
            ),
            (
                "Attempt standard remediation: power-cycle, reconnect, driver/firmware update.",
                "Basic remediation attempted",
            ),
            (
                "Check supporting infrastructure (print server, network port, power).",
                "Infrastructure checked",
            ),
            ("Swap failed hardware from spares where remediation fails.", "Device restored"),
            ("Log the fault and order replacements to replenish spares.", None),
        ),
        escalation_team="IT-Helpdesk",
        severity_hint=Severity.SEV4,
        estimated_resolution_minutes=45,
        tags=("hardware", "peripheral", "end-user"),
    ),
)


def build_runbooks() -> tuple[Runbook, ...]:
    """Build the full runbook catalog (one runbook per subcategory)."""

    counters: dict[Category, int] = {}
    id_by_sub: dict[Subcategory, str] = {}
    for spec in _RUNBOOK_SPECS:
        cat = category_of(spec.subcategory)
        counters[cat] = counters.get(cat, 0) + 1
        id_by_sub[spec.subcategory] = f"RB-{cat.value}-{counters[cat]:03d}"

    runbooks: list[Runbook] = []
    for spec in _RUNBOOK_SPECS:
        cat = category_of(spec.subcategory)
        steps = tuple(
            RunbookStep(order=i, action=action, expected_signal=expected)
            for i, (action, expected) in enumerate(spec.steps, start=1)
        )
        related = tuple(id_by_sub[s] for s in spec.related if s in id_by_sub)
        runbooks.append(
            Runbook(
                runbook_id=id_by_sub[spec.subcategory],
                title=spec.title,
                category=cat,
                subcategory=spec.subcategory,
                summary=spec.summary,
                symptoms=spec.symptoms,
                steps=steps,
                escalation_team=spec.escalation_team,
                severity_hint=spec.severity_hint,
                estimated_resolution_minutes=spec.estimated_resolution_minutes,
                tags=spec.tags,
                last_reviewed=LAST_REVIEWED,
                related_runbook_ids=related,
            )
        )
    return tuple(runbooks)


__all__ = ["LAST_REVIEWED", "build_runbooks"]
