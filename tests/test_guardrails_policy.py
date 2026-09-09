"""Tests for GuardrailPolicyConfig."""

from __future__ import annotations

import pytest

from nirnayax.guardrails import GuardrailPolicyConfig


def test_guardrail_policy_defaults() -> None:
    policy = GuardrailPolicyConfig()
    assert policy.fail_closed is True
    assert policy.min_confidence_threshold == 0.50
    assert policy.prohibit_auto_close_critical is True
    assert "check_service_metrics" in policy.allowed_tools
    assert "reset_bgp_peering_session" in policy.allowed_remediation_actions


def test_guardrail_policy_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIRNAYAX_GUARDRAILS_FAIL_CLOSED", "false")
    monkeypatch.setenv("NIRNAYAX_MIN_CONFIDENCE_THRESHOLD", "0.75")
    policy = GuardrailPolicyConfig.from_env()

    assert policy.fail_closed is False
    assert policy.min_confidence_threshold == 0.75
