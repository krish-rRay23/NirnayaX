"""Static reference data for synthetic incident generation.

This module holds the *content* the generator draws from — realistic telecom/OSS
service names, datacenter regions, per-category channel mixes, and one
:class:`IncidentTemplate` per subcategory. Keeping it separate from
:mod:`nirnayax.data.generator` means the generation *logic* stays small and the
domain *flavour* can be tuned or extended without touching that logic.

Every template guarantees that the placeholders used in its ``titles`` /
``descriptions`` are exactly ``{service}``, ``{region}``, ``{host}`` plus the
names of its own signals — this invariant is asserted by the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.taxonomy import Category, Channel, Severity, Subcategory


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Specification for a numeric telemetry signal to sample."""

    name: str
    unit: str
    low: float
    high: float
    integer: bool = False


@dataclass(frozen=True, slots=True)
class IncidentTemplate:
    """Everything needed to synthesise incidents for one subcategory."""

    subcategory: Subcategory
    services: tuple[str, ...]
    titles: tuple[str, ...]
    descriptions: tuple[str, ...]
    signals: tuple[SignalSpec, ...]
    severity_weights: tuple[tuple[Severity, float], ...]
    tags: tuple[str, ...]
    customer_impacting_rate: float = 0.4


# Datacenter / NOC regions (telecom flavour).
REGIONS: tuple[str, ...] = (
    "DC-BLR-01",
    "DC-MUM-02",
    "DC-DEL-01",
    "DC-CHN-01",
    "DC-HYD-01",
    "DC-KOL-01",
    "EDGE-PUN-01",
    "EDGE-AMD-01",
)

# Realistic category prior: application/DB and network dominate the L1 queue.
CATEGORY_WEIGHTS: tuple[tuple[Category, float], ...] = (
    (Category.NETWORK, 0.30),
    (Category.APPLICATION_DB, 0.35),
    (Category.BILLING_OSS, 0.20),
    (Category.HARDWARE_ACCESS, 0.15),
)

# How incidents in each category tend to arrive.
CATEGORY_CHANNEL_WEIGHTS: dict[Category, tuple[tuple[Channel, float], ...]] = {
    Category.NETWORK: (
        (Channel.MONITORING, 0.85),
        (Channel.EMAIL, 0.08),
        (Channel.PHONE, 0.05),
        (Channel.CHAT, 0.02),
    ),
    Category.APPLICATION_DB: (
        (Channel.MONITORING, 0.80),
        (Channel.EMAIL, 0.10),
        (Channel.CHAT, 0.06),
        (Channel.PHONE, 0.04),
    ),
    Category.BILLING_OSS: (
        (Channel.MONITORING, 0.55),
        (Channel.EMAIL, 0.25),
        (Channel.CHAT, 0.10),
        (Channel.PHONE, 0.10),
    ),
    Category.HARDWARE_ACCESS: (
        (Channel.PHONE, 0.40),
        (Channel.SELF_SERVICE, 0.25),
        (Channel.EMAIL, 0.20),
        (Channel.CHAT, 0.10),
        (Channel.MONITORING, 0.05),
    ),
}

# Common severity mixes reused across templates.
_SEV_CRITICAL = ((Severity.SEV1, 0.45), (Severity.SEV2, 0.40), (Severity.SEV3, 0.15))
_SEV_HIGH = ((Severity.SEV2, 0.50), (Severity.SEV3, 0.35), (Severity.SEV1, 0.15))
_SEV_MEDIUM = ((Severity.SEV3, 0.55), (Severity.SEV2, 0.30), (Severity.SEV4, 0.15))
_SEV_LOW = ((Severity.SEV4, 0.55), (Severity.SEV3, 0.35), (Severity.SEV2, 0.10))


SUBCATEGORY_TEMPLATES: dict[Subcategory, IncidentTemplate] = {
    # ---------------------------- NETWORK ----------------------------
    Subcategory.LATENCY_PACKET_LOSS: IncidentTemplate(
        subcategory=Subcategory.LATENCY_PACKET_LOSS,
        services=(
            "Core-Router-Edge",
            "Access-Gateway",
            "MPLS-Backbone",
            "SD-WAN-Controller",
            "Metro-Ethernet-Ring",
        ),
        titles=(
            "Elevated packet loss on {service} in {region}",
            "High latency and packet loss affecting {service} ({region})",
            "{service} reporting {packet_loss_pct}% packet loss in {region}",
        ),
        descriptions=(
            "Monitoring detected {packet_loss_pct}% packet loss and RTT of {rtt_ms} ms on {service} ({host}) in {region}. Jitter measured at {jitter_ms} ms; customer traffic degradation reported.",
            "Sustained latency of {rtt_ms} ms with {packet_loss_pct}% loss on {service} in {region}. Suspected congestion or a faulty upstream link on {host}.",
        ),
        signals=(
            SignalSpec("packet_loss_pct", "%", 1.0, 25.0),
            SignalSpec("rtt_ms", "ms", 40, 600, integer=True),
            SignalSpec("jitter_ms", "ms", 5, 120, integer=True),
        ),
        severity_weights=_SEV_HIGH,
        tags=("network", "latency", "packet-loss"),
        customer_impacting_rate=0.6,
    ),
    Subcategory.LINK_DOWN: IncidentTemplate(
        subcategory=Subcategory.LINK_DOWN,
        services=(
            "MPLS-Backbone",
            "Metro-Ethernet-Ring",
            "Access-Gateway",
            "Core-Router-Edge",
            "Fiber-Uplink",
        ),
        titles=(
            "Link down on {service} in {region}",
            "{links_down} interface(s) down on {service} ({region})",
            "Circuit outage detected on {service} in {region}",
        ),
        descriptions=(
            "{links_down} physical interface(s) reported DOWN on {service} ({host}) in {region}. Redundant paths remaining: {redundancy_remaining}. Automatic failover status uncertain.",
            "Loss of carrier on {service} in {region}; {links_down} link(s) down on {host}. Field engineering may be required.",
        ),
        signals=(
            SignalSpec("links_down", "count", 1, 6, integer=True),
            SignalSpec("redundancy_remaining", "count", 0, 2, integer=True),
        ),
        severity_weights=_SEV_CRITICAL,
        tags=("network", "link", "outage"),
        customer_impacting_rate=0.8,
    ),
    Subcategory.DNS_RESOLUTION: IncidentTemplate(
        subcategory=Subcategory.DNS_RESOLUTION,
        services=("DNS-Resolver", "Recursive-DNS", "Authoritative-DNS", "Service-Discovery"),
        titles=(
            "DNS resolution failures on {service} in {region}",
            "{resolution_failure_pct}% DNS query failures via {service} ({region})",
        ),
        descriptions=(
            "{service} ({host}) in {region} returning SERVFAIL for {resolution_failure_pct}% of queries; median resolution latency {query_latency_ms} ms. Downstream services failing to resolve endpoints.",
            "Elevated DNS resolution errors ({resolution_failure_pct}%) on {service} in {region}. Query latency spiked to {query_latency_ms} ms on {host}.",
        ),
        signals=(
            SignalSpec("resolution_failure_pct", "%", 5.0, 100.0),
            SignalSpec("query_latency_ms", "ms", 50, 3000, integer=True),
        ),
        severity_weights=_SEV_HIGH,
        tags=("network", "dns"),
        customer_impacting_rate=0.5,
    ),
    Subcategory.BGP_ROUTING: IncidentTemplate(
        subcategory=Subcategory.BGP_ROUTING,
        services=("Core-Router-Edge", "MPLS-Backbone", "Peering-Router", "Route-Reflector"),
        titles=(
            "BGP session flapping on {service} in {region}",
            "Route withdrawals detected on {service} ({region})",
        ),
        descriptions=(
            "{bgp_sessions_flapping} BGP session(s) flapping on {service} ({host}) in {region}; approximately {routes_withdrawn} routes withdrawn. Possible reachability impact to peered networks.",
            "Routing instability on {service} in {region}: {bgp_sessions_flapping} peering sessions resetting, {routes_withdrawn} prefixes affected on {host}.",
        ),
        signals=(
            SignalSpec("bgp_sessions_flapping", "count", 1, 8, integer=True),
            SignalSpec("routes_withdrawn", "count", 50, 50000, integer=True),
        ),
        severity_weights=_SEV_HIGH,
        tags=("network", "bgp", "routing"),
        customer_impacting_rate=0.6,
    ),
    # ------------------------- APPLICATION_DB -------------------------
    Subcategory.CONNECTION_POOL_EXHAUSTION: IncidentTemplate(
        subcategory=Subcategory.CONNECTION_POOL_EXHAUSTION,
        services=(
            "OrderMgmt-API",
            "CRM-Portal",
            "CustomerProfile-Service",
            "Notification-Service",
            "Billing-API",
        ),
        titles=(
            "Connection pool exhaustion on {service} in {region}",
            "{service} database connection pool saturated ({region})",
        ),
        descriptions=(
            "{service} ({host}) in {region} hitting {pool_utilization_pct}% DB connection-pool utilization; new checkouts waiting {connection_wait_ms} ms and {error_rate_pct}% of requests failing with pool-timeout errors.",
            "Connection pool on {service} exhausted in {region}. Utilization {pool_utilization_pct}%, checkout wait {connection_wait_ms} ms on {host}; threads blocked awaiting connections.",
        ),
        signals=(
            SignalSpec("pool_utilization_pct", "%", 96.0, 100.0),
            SignalSpec("connection_wait_ms", "ms", 500, 20000, integer=True),
            SignalSpec("error_rate_pct", "%", 2.0, 60.0),
        ),
        severity_weights=_SEV_HIGH,
        tags=("application", "database", "connection-pool"),
        customer_impacting_rate=0.7,
    ),
    Subcategory.REPLICATION_LAG: IncidentTemplate(
        subcategory=Subcategory.REPLICATION_LAG,
        services=(
            "Billing-DB",
            "Inventory-DB",
            "CustomerProfile-DB",
            "Reporting-Replica",
            "Analytics-Replica",
        ),
        titles=(
            "Replication lag on {service} in {region}",
            "{service} read replica lagging by {replica_lag_seconds}s ({region})",
        ),
        descriptions=(
            "Read replica for {service} ({host}) in {region} lagging the primary by {replica_lag_seconds} seconds; WAL/redo backlog ~{wal_backlog_mb} MB. Stale reads likely for downstream consumers.",
            "Replication lag on {service} in {region} reached {replica_lag_seconds}s ({wal_backlog_mb} MB backlog on {host}). Failover risk if the primary is lost.",
        ),
        signals=(
            SignalSpec("replica_lag_seconds", "s", 30, 3600, integer=True),
            SignalSpec("wal_backlog_mb", "MB", 100, 20000, integer=True),
        ),
        severity_weights=_SEV_MEDIUM,
        tags=("application", "database", "replication"),
        customer_impacting_rate=0.4,
    ),
    Subcategory.DEADLOCK: IncidentTemplate(
        subcategory=Subcategory.DEADLOCK,
        services=("OrderMgmt-DB", "Billing-DB", "Inventory-DB", "Payments-DB"),
        titles=(
            "Database deadlocks on {service} in {region}",
            "{deadlocks_per_min} deadlocks/min on {service} ({region})",
        ),
        descriptions=(
            "{service} ({host}) in {region} logging {deadlocks_per_min} deadlocks per minute; {aborted_txns} transactions aborted, lock waits up to {lock_wait_ms} ms. Application retries and rollbacks climbing.",
            "Deadlock storm on {service} in {region}: {deadlocks_per_min}/min, {aborted_txns} aborted transactions on {host}.",
        ),
        signals=(
            SignalSpec("deadlocks_per_min", "count", 3, 120, integer=True),
            SignalSpec("aborted_txns", "count", 5, 500, integer=True),
            SignalSpec("lock_wait_ms", "ms", 500, 30000, integer=True),
        ),
        severity_weights=_SEV_HIGH,
        tags=("application", "database", "deadlock", "concurrency"),
        customer_impacting_rate=0.55,
    ),
    Subcategory.SLOW_QUERY: IncidentTemplate(
        subcategory=Subcategory.SLOW_QUERY,
        services=("Reporting-DB", "Analytics-DB", "CRM-Portal", "OrderMgmt-DB", "Search-Service"),
        titles=(
            "Slow queries degrading {service} in {region}",
            "p99 query latency {p99_query_ms} ms on {service} ({region})",
        ),
        descriptions=(
            "{service} ({host}) in {region} p99 query latency at {p99_query_ms} ms; {slow_query_pct}% of {qps} qps exceeding SLO. Likely a missing index or plan regression.",
            "Query performance degraded on {service} in {region}: p99 {p99_query_ms} ms, {slow_query_pct}% slow queries at {qps} qps on {host}.",
        ),
        signals=(
            SignalSpec("p99_query_ms", "ms", 1500, 45000, integer=True),
            SignalSpec("qps", "count", 50, 5000, integer=True),
            SignalSpec("slow_query_pct", "%", 5.0, 80.0),
        ),
        severity_weights=_SEV_MEDIUM,
        tags=("application", "database", "slow-query", "performance"),
        customer_impacting_rate=0.5,
    ),
    Subcategory.DISK_SPACE: IncidentTemplate(
        subcategory=Subcategory.DISK_SPACE,
        services=("Billing-DB", "Inventory-DB", "Log-Aggregator", "OrderMgmt-DB", "Backup-Store"),
        titles=(
            "Low disk space on {service} in {region}",
            "{service} volume {disk_used_pct}% full in {region}",
        ),
        descriptions=(
            "Data volume on {service} ({host}) in {region} at {disk_used_pct}% used ({free_gb} GB free); inode usage {inode_used_pct}%. Write failures imminent if not remediated.",
            "Disk space critical on {service} in {region}: {disk_used_pct}% used, {free_gb} GB remaining on {host}. Rapid WAL/log growth suspected.",
        ),
        signals=(
            SignalSpec("disk_used_pct", "%", 90.0, 100.0),
            SignalSpec("free_gb", "GB", 0, 40, integer=True),
            SignalSpec("inode_used_pct", "%", 60.0, 100.0),
        ),
        severity_weights=_SEV_HIGH,
        tags=("application", "database", "disk", "capacity"),
        customer_impacting_rate=0.45,
    ),
    # -------------------------- BILLING_OSS ---------------------------
    Subcategory.RATING_ENGINE_ERROR: IncidentTemplate(
        subcategory=Subcategory.RATING_ENGINE_ERROR,
        services=("Rating-Engine", "Charging-System", "Tariff-Service", "Online-Charging"),
        titles=(
            "Rating engine errors on {service} in {region}",
            "{service} failing to rate CDRs ({region})",
        ),
        descriptions=(
            "{service} ({host}) in {region} failing to rate usage records; {cdr_rating_failures} CDRs errored ({error_rate_pct}% of the batch). Revenue-leakage risk if unresolved.",
            "Rating errors on {service} in {region}: {error_rate_pct}% of events unrated, {cdr_rating_failures} records queued on {host}.",
        ),
        signals=(
            SignalSpec("cdr_rating_failures", "count", 100, 500000, integer=True),
            SignalSpec("error_rate_pct", "%", 1.0, 40.0),
        ),
        severity_weights=_SEV_MEDIUM,
        tags=("billing", "oss", "rating", "revenue"),
        customer_impacting_rate=0.3,
    ),
    Subcategory.INVOICE_GENERATION_FAILURE: IncidentTemplate(
        subcategory=Subcategory.INVOICE_GENERATION_FAILURE,
        services=("Invoicing-Service", "Billing-Batch", "Statement-Generator", "Document-Service"),
        titles=(
            "Invoice generation failure on {service} in {region}",
            "{invoices_failed} invoices failed on {service} ({region})",
        ),
        descriptions=(
            "Nightly invoicing batch on {service} ({host}) in {region} failed for {invoices_failed} accounts; batch delayed {batch_delay_min} minutes. Bill-cycle SLA at risk.",
            "{service} in {region} could not generate {invoices_failed} invoices ({batch_delay_min} minutes behind schedule on {host}).",
        ),
        signals=(
            SignalSpec("invoices_failed", "count", 10, 200000, integer=True),
            SignalSpec("batch_delay_min", "min", 15, 1440, integer=True),
        ),
        severity_weights=_SEV_MEDIUM,
        tags=("billing", "oss", "invoicing", "batch"),
        customer_impacting_rate=0.35,
    ),
    Subcategory.MEDIATION_FEED_GAP: IncidentTemplate(
        subcategory=Subcategory.MEDIATION_FEED_GAP,
        services=("Mediation-Gateway", "CDR-Collector", "Usage-Mediation", "Event-Ingestor"),
        titles=(
            "Mediation feed gap on {service} in {region}",
            "Missing usage feeds on {service} ({region})",
        ),
        descriptions=(
            "{service} ({host}) in {region} missing {missing_files} mediation file(s); feed delayed {feed_delay_min} minutes and ~{dropped_records} usage records not ingested. Downstream rating starved.",
            "Usage mediation gap on {service} in {region}: {missing_files} files absent, {feed_delay_min} min delay on {host}.",
        ),
        signals=(
            SignalSpec("missing_files", "count", 1, 240, integer=True),
            SignalSpec("feed_delay_min", "min", 20, 2880, integer=True),
            SignalSpec("dropped_records", "count", 100, 2000000, integer=True),
        ),
        severity_weights=_SEV_MEDIUM,
        tags=("billing", "oss", "mediation", "usage"),
        customer_impacting_rate=0.25,
    ),
    Subcategory.PROVISIONING_SYNC_FAILURE: IncidentTemplate(
        subcategory=Subcategory.PROVISIONING_SYNC_FAILURE,
        services=(
            "Provisioning-Orchestrator",
            "Order-Fulfillment",
            "Activation-Service",
            "Network-Provisioning",
        ),
        titles=(
            "Provisioning sync failure on {service} in {region}",
            "{orders_stuck} orders stuck in {service} ({region})",
        ),
        descriptions=(
            "{service} ({host}) in {region} failing to sync activations to the network; {orders_stuck} orders stuck after {sync_retries} retries. New customer activations blocked.",
            "Provisioning synchronization failing on {service} in {region}: {orders_stuck} orders pending, {sync_retries} retries exhausted on {host}.",
        ),
        signals=(
            SignalSpec("orders_stuck", "count", 5, 50000, integer=True),
            SignalSpec("sync_retries", "count", 3, 200, integer=True),
        ),
        severity_weights=_SEV_HIGH,
        tags=("billing", "oss", "provisioning", "activation"),
        customer_impacting_rate=0.5,
    ),
    # ------------------------ HARDWARE_ACCESS -------------------------
    Subcategory.ACCOUNT_LOCKOUT: IncidentTemplate(
        subcategory=Subcategory.ACCOUNT_LOCKOUT,
        services=("IdentityProvider", "ActiveDirectory", "SSO-Gateway", "IAM-Service"),
        titles=(
            "Account lockouts reported via {service} in {region}",
            "{affected_users} users locked out ({service}, {region})",
        ),
        descriptions=(
            "{affected_users} user account(s) locked out on {service} ({host}) in {region} after repeated failed logins ({failed_logins} attempts observed). Possible policy misconfiguration or brute-force lockout.",
            "Users unable to authenticate via {service} in {region}; {affected_users} accounts locked, {failed_logins} failed attempts on {host}.",
        ),
        signals=(
            SignalSpec("affected_users", "count", 1, 2000, integer=True),
            SignalSpec("failed_logins", "count", 3, 500, integer=True),
        ),
        severity_weights=_SEV_MEDIUM,
        tags=("access", "identity", "lockout"),
        customer_impacting_rate=0.4,
    ),
    Subcategory.VPN_ACCESS_FAILURE: IncidentTemplate(
        subcategory=Subcategory.VPN_ACCESS_FAILURE,
        services=("VPN-Concentrator", "Remote-Access-Gateway", "SSL-VPN", "ZTNA-Gateway"),
        titles=(
            "VPN access failures on {service} in {region}",
            "{affected_users} users unable to connect via {service} ({region})",
        ),
        descriptions=(
            "{service} ({host}) in {region} rejecting VPN connections; {affected_users} users affected with {tunnel_failures} tunnel-setup failures. Remote workforce impacted.",
            "VPN access degraded on {service} in {region}: {affected_users} users unable to establish tunnels ({tunnel_failures} failures on {host}).",
        ),
        signals=(
            SignalSpec("affected_users", "count", 1, 3000, integer=True),
            SignalSpec("tunnel_failures", "count", 5, 800, integer=True),
        ),
        severity_weights=_SEV_HIGH,
        tags=("access", "vpn", "remote"),
        customer_impacting_rate=0.3,
    ),
    Subcategory.SERVER_HARDWARE_FAULT: IncidentTemplate(
        subcategory=Subcategory.SERVER_HARDWARE_FAULT,
        services=("BladeServer", "RackServer", "StorageArray", "HypervisorHost"),
        titles=(
            "Hardware fault on {service} in {region}",
            "{service} reporting component failure in {region}",
        ),
        descriptions=(
            "{service} ({host}) in {region} raised a hardware alarm: {failed_components} component(s) failed, chassis temperature {temperature_c} C. Redundancy degraded; field replacement may be needed.",
            "Hardware fault on {service} ({host}) in {region}: {failed_components} failed component(s), temperature {temperature_c} C.",
        ),
        signals=(
            SignalSpec("failed_components", "count", 1, 6, integer=True),
            SignalSpec("temperature_c", "C", 60, 95, integer=True),
        ),
        severity_weights=_SEV_HIGH,
        tags=("hardware", "server", "fault"),
        customer_impacting_rate=0.35,
    ),
    Subcategory.PERIPHERAL_FAILURE: IncidentTemplate(
        subcategory=Subcategory.PERIPHERAL_FAILURE,
        services=(
            "Workstation",
            "Printer-Fleet",
            "Badge-Reader",
            "VoIP-Handset",
            "Conference-System",
        ),
        titles=(
            "Peripheral failure affecting {service} in {region}",
            "{affected_devices} devices down: {service} ({region})",
        ),
        descriptions=(
            "{affected_devices} {service} device(s) reported non-functional in {region} (site {host}). End users unable to complete workflow; hardware or driver fault suspected.",
            "Peripheral/end-user hardware failure in {region}: {affected_devices} {service} unit(s) affected at {host}.",
        ),
        signals=(SignalSpec("affected_devices", "count", 1, 120, integer=True),),
        severity_weights=_SEV_LOW,
        tags=("hardware", "peripheral", "end-user"),
        customer_impacting_rate=0.15,
    ),
}

# Placeholders always available to a template's format context.
BASE_PLACEHOLDERS: frozenset[str] = frozenset({"service", "region", "host"})

__all__ = [
    "BASE_PLACEHOLDERS",
    "CATEGORY_CHANNEL_WEIGHTS",
    "CATEGORY_WEIGHTS",
    "REGIONS",
    "SUBCATEGORY_TEMPLATES",
    "IncidentTemplate",
    "SignalSpec",
]
