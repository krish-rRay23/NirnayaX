"""Tests for simulated diagnostic and remediation tools."""

from __future__ import annotations

from nirnayax.agent.tools import (
    SimulatedEnvironment,
    check_service_metrics,
    execute_remediation,
    fetch_service_logs,
    verify_service_recovery,
)


def test_simulated_environment_initialization_and_anomaly() -> None:
    env = SimulatedEnvironment({"db-service": "CONNECTION_POOL_EXHAUSTION"})
    obs = check_service_metrics(env, "db-service")
    assert not obs.passed
    assert obs.metric_name == "pool_utilization_pct"
    assert obs.value == 100.0


def test_fetch_service_logs() -> None:
    env = SimulatedEnvironment({"app-service": "DISK_SPACE"})
    logs = fetch_service_logs(env, "app-service")
    assert logs.tool_name == "log_analyzer"
    assert logs.target_service == "app-service"
    assert logs.value > 0.0


def test_remediation_and_verification_flow() -> None:
    service = "core-router"
    subcat = "BGP_ROUTING"
    env = SimulatedEnvironment({service: subcat})

    # 1. Observe anomaly
    obs_before = check_service_metrics(env, service)
    assert not obs_before.passed

    # 2. Act remediation
    action = execute_remediation(env, service, subcat)
    assert action.success
    assert action.action_taken == "reset_bgp_peering_session"

    # 3. Verify recovery
    obs_after = verify_service_recovery(env, service, subcat)
    assert obs_after.passed
    assert obs_after.value >= 90.0  # Normal health score (100.0)
