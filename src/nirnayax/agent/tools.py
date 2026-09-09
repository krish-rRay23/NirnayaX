"""Deterministic simulated diagnostic and remediation tools (observe -> act -> verify)."""

from __future__ import annotations

from .types import Observation, RemediationAction


class ServiceState:
    """Mutable state for a single service in the simulated environment."""

    def __init__(self, service: str, initial_subcategory: str | None = None) -> None:
        self.service = service
        self.anomaly = initial_subcategory
        # Set metrics based on anomaly
        if initial_subcategory == "CONNECTION_POOL_EXHAUSTION":
            self.active_connections = 100
            self.max_connections = 100
            self.disk_used_pct = 45.0
            self.bgp_flaps = 0
            self.error_rate = 35.0
        elif initial_subcategory == "DISK_SPACE":
            self.active_connections = 25
            self.max_connections = 100
            self.disk_used_pct = 96.5
            self.bgp_flaps = 0
            self.error_rate = 12.0
        elif initial_subcategory == "BGP_ROUTING":
            self.active_connections = 20
            self.max_connections = 100
            self.disk_used_pct = 40.0
            self.bgp_flaps = 24
            self.error_rate = 28.0
        elif initial_subcategory == "DNS_RESOLUTION":
            self.active_connections = 15
            self.max_connections = 100
            self.disk_used_pct = 30.0
            self.bgp_flaps = 0
            self.error_rate = 40.0
        else:
            # Healthy default
            self.active_connections = 20
            self.max_connections = 100
            self.disk_used_pct = 45.0
            self.bgp_flaps = 0
            self.error_rate = 0.5


class SimulatedEnvironment:
    """State container for simulated infrastructure services."""

    def __init__(self, service_anomalies: dict[str, str] | None = None) -> None:
        self._services: dict[str, ServiceState] = {}
        if service_anomalies:
            for service, subcat in service_anomalies.items():
                self._services[service] = ServiceState(service, subcat)

    def get_service(self, service: str, subcategory_hint: str | None = None) -> ServiceState:
        if service not in self._services:
            self._services[service] = ServiceState(service, subcategory_hint)
        return self._services[service]


def check_service_metrics(
    env: SimulatedEnvironment, service: str, subcategory_hint: str | None = None
) -> Observation:
    """Observe service metrics to detect specific failure anomalies."""

    state = env.get_service(service, subcategory_hint)

    is_conn_pool = (
        state.anomaly == "CONNECTION_POOL_EXHAUSTION"
        or (
            subcategory_hint == "CONNECTION_POOL_EXHAUSTION"
            and state.active_connections >= state.max_connections
        )
    )
    if is_conn_pool:
        return Observation(
            tool_name="metrics_monitor",
            target_service=service,
            metric_name="pool_utilization_pct",
            value=(state.active_connections / state.max_connections) * 100.0,
            passed=False,
            details="Connection pool exhausted (100% allocated)",
        )
    elif state.anomaly == "DISK_SPACE" or (
        subcategory_hint == "DISK_SPACE" and state.disk_used_pct > 90.0
    ):
        return Observation(
            tool_name="metrics_monitor",
            target_service=service,
            metric_name="disk_used_pct",
            value=state.disk_used_pct,
            passed=False,
            details=f"Disk usage critically high ({state.disk_used_pct:.1f}%)",
        )
    elif state.anomaly == "BGP_ROUTING" or (
        subcategory_hint == "BGP_ROUTING" and state.bgp_flaps > 5
    ):
        return Observation(
            tool_name="metrics_monitor",
            target_service=service,
            metric_name="bgp_flaps_per_min",
            value=float(state.bgp_flaps),
            passed=False,
            details=f"BGP session unstable ({state.bgp_flaps} flaps/min)",
        )
    elif state.anomaly == "DNS_RESOLUTION" or (
        subcategory_hint == "DNS_RESOLUTION" and state.error_rate > 20.0
    ):
        return Observation(
            tool_name="metrics_monitor",
            target_service=service,
            metric_name="dns_error_pct",
            value=state.error_rate,
            passed=False,
            details=f"DNS resolution failure rate high ({state.error_rate:.1f}%)",
        )

    # General / healthy check
    is_healthy = (
        state.active_connections < state.max_connections
        and state.disk_used_pct <= 90.0
        and state.bgp_flaps <= 5
        and state.error_rate <= 10.0
    )
    details = (
        "Metrics operating within normal thresholds"
        if is_healthy
        else "Sub-optimal metrics detected"
    )
    return Observation(
        tool_name="metrics_monitor",
        target_service=service,
        metric_name="service_health_score",
        value=100.0 if is_healthy else 30.0,
        passed=is_healthy,
        details=details,
    )


def fetch_service_logs(env: SimulatedEnvironment, service: str) -> Observation:
    """Observe recent error log occurrences for a service."""

    state = env.get_service(service)
    passed = state.error_rate < 5.0
    return Observation(
        tool_name="log_analyzer",
        target_service=service,
        metric_name="error_log_count",
        value=state.error_rate * 2.0,
        passed=passed,
        details=f"Extracted {int(state.error_rate * 2)} error log events in last 15m",
    )


def execute_remediation(
    env: SimulatedEnvironment, service: str, subcategory: str
) -> RemediationAction:
    """Act: Execute deterministic simulated remediation tool for the subcategory."""

    state = env.get_service(service, subcategory)

    if subcategory == "CONNECTION_POOL_EXHAUSTION":
        state.active_connections = 15
        state.error_rate = 0.0
        state.anomaly = None
        action = "restart_connection_pool"
        details = "Flushed idle connections and resized pool buffer."
    elif subcategory == "DISK_SPACE":
        state.disk_used_pct = 42.0
        state.error_rate = 0.0
        state.anomaly = None
        action = "purge_temp_logs"
        details = "Truncated old application log files and temporary trace files."
    elif subcategory == "BGP_ROUTING":
        state.bgp_flaps = 0
        state.error_rate = 0.0
        state.anomaly = None
        action = "reset_bgp_peering_session"
        details = "Hard reset BGP neighbor state and re-established peering."
    elif subcategory == "DNS_RESOLUTION":
        state.error_rate = 0.0
        state.anomaly = None
        action = "flush_dns_cache_and_reload"
        details = "Flushed local DNS cache and reloaded resolver configuration."
    else:
        # Fallback default remediation
        state.active_connections = 20
        state.disk_used_pct = 45.0
        state.bgp_flaps = 0
        state.error_rate = 0.0
        state.anomaly = None
        action = f"restart_service_{service}"
        details = f"Gracefully restarted service container {service}."

    return RemediationAction(
        tool_name="remediation_executor",
        target_service=service,
        action_taken=action,
        success=True,
        details=details,
    )


def verify_service_recovery(
    env: SimulatedEnvironment, service: str, subcategory: str
) -> Observation:
    """Verify: Re-observe service metrics after remediation to confirm recovery."""

    obs = check_service_metrics(env, service, subcategory)
    return Observation(
        tool_name="recovery_verifier",
        target_service=service,
        metric_name=obs.metric_name,
        value=obs.value,
        passed=obs.passed,
        details=f"Verification: {obs.details}",
    )


__all__ = [
    "ServiceState",
    "SimulatedEnvironment",
    "check_service_metrics",
    "execute_remediation",
    "fetch_service_logs",
    "verify_service_recovery",
]
