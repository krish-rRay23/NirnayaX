"""Tests for human approval gate and risk assessment."""

from __future__ import annotations

import pytest

from nirnayax.agent.approval import (
    create_approval_request,
    evaluate_remediation_risk,
    process_approval_decision,
)
from nirnayax.jira.types import ApprovalRiskLevel, ApprovalStatus


def test_evaluate_remediation_risk_levels() -> None:
    sev1_risk = evaluate_remediation_risk("SEV1", "BGP_ROUTING", "reset_bgp")
    crit_risk = evaluate_remediation_risk("Critical", "DISK_SPACE", "purge_logs")
    sev2_risk = evaluate_remediation_risk("SEV2", "BGP_ROUTING", "reset_bgp")
    assert sev1_risk == ApprovalRiskLevel.CRITICAL
    assert crit_risk == ApprovalRiskLevel.CRITICAL
    assert sev2_risk == ApprovalRiskLevel.HIGH
    assert (
        evaluate_remediation_risk("SEV2", "CONNECTION_POOL_EXHAUSTION", "restart_pool")
        == ApprovalRiskLevel.MEDIUM
    )
    assert evaluate_remediation_risk("SEV3", "OTHER", "restart_app") == ApprovalRiskLevel.LOW


def test_critical_and_sev1_never_auto_approve() -> None:
    req = create_approval_request(
        request_id="REQ-1",
        issue_key="INC-101",
        severity="SEV1",
        subcategory="OTHER",
        proposed_action="low_risk_action",
        reasoning="Testing SEV1 strict rule",
        allow_auto_approve_low_risk=True,  # Even with auto-approve enabled
    )

    assert req.risk_level == ApprovalRiskLevel.CRITICAL
    assert req.status == ApprovalStatus.PENDING  # Explicit approval REQUIRED


def test_high_risk_action_requires_explicit_approval() -> None:
    req = create_approval_request(
        request_id="REQ-2",
        issue_key="INC-102",
        severity="SEV2",
        subcategory="BGP_ROUTING",
        proposed_action="reset_bgp_peering_session",
        reasoning="High risk action",
        allow_auto_approve_low_risk=True,
    )

    assert req.risk_level == ApprovalRiskLevel.HIGH
    assert req.status == ApprovalStatus.PENDING


def test_low_risk_auto_approval_when_enabled() -> None:
    req = create_approval_request(
        request_id="REQ-3",
        issue_key="INC-103",
        severity="SEV3",
        subcategory="OTHER",
        proposed_action="minor_cleanup",
        reasoning="Low risk",
        allow_auto_approve_low_risk=True,
    )

    assert req.risk_level == ApprovalRiskLevel.LOW
    assert req.status == ApprovalStatus.AUTO_APPROVED


def test_process_approval_decision_approve_and_reject() -> None:
    req = create_approval_request(
        request_id="REQ-4",
        issue_key="INC-104",
        severity="SEV2",
        subcategory="BGP_ROUTING",
        proposed_action="reset_bgp_peering_session",
        reasoning="Test decision",
    )

    approved = process_approval_decision(req, ApprovalStatus.APPROVED, approved_by="sr-engineer")
    assert approved.status == ApprovalStatus.APPROVED
    assert approved.approved_by == "sr-engineer"
    assert approved.approved_at is not None

    rejected = process_approval_decision(req, ApprovalStatus.REJECTED, approved_by="on-call-lead")
    assert rejected.status == ApprovalStatus.REJECTED
    assert rejected.approved_by == "on-call-lead"


def test_process_approval_decision_invalid_raises() -> None:
    req = create_approval_request(
        request_id="REQ-5",
        issue_key="INC-105",
        severity="SEV2",
        subcategory="OTHER",
        proposed_action="action",
        reasoning="Test invalid",
    )
    with pytest.raises(ValueError, match="Invalid approval decision"):
        process_approval_decision(req, ApprovalStatus.PENDING)
