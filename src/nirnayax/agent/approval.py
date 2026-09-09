"""Human approval gate and risk assessment for remediation actions."""

from __future__ import annotations

from datetime import UTC, datetime

from ..jira.types import ApprovalRequest, ApprovalRiskLevel, ApprovalStatus


def evaluate_remediation_risk(
    severity: str, subcategory: str, proposed_action: str
) -> ApprovalRiskLevel:
    """Assess the operational risk level of a proposed remediation action."""

    sev_upper = severity.upper()
    if sev_upper in ("SEV1", "CRITICAL", "P1"):
        return ApprovalRiskLevel.CRITICAL

    subcat_upper = subcategory.upper()
    if subcat_upper in ("BGP_ROUTING", "SERVER_HARDWARE_FAULT") or "bgp" in proposed_action.lower():
        return ApprovalRiskLevel.HIGH

    if subcat_upper in ("CONNECTION_POOL_EXHAUSTION", "DISK_SPACE", "DNS_RESOLUTION"):
        return ApprovalRiskLevel.MEDIUM

    return ApprovalRiskLevel.LOW


def create_approval_request(
    *,
    request_id: str,
    issue_key: str,
    severity: str,
    subcategory: str,
    proposed_action: str,
    reasoning: str,
    allow_auto_approve_low_risk: bool = False,
) -> ApprovalRequest:
    """Create an approval request enforcing strict rules for Critical/SEV1 incidents.

    Rule: Critical/SEV1 incidents or HIGH/CRITICAL risk actions NEVER auto-approve.
    Approval must be explicit; Critical/SEV1 incidents are never auto-closed.
    """

    risk = evaluate_remediation_risk(severity, subcategory, proposed_action)

    # Critical / SEV1 incidents or High/Critical risk actions NEVER auto-approve.
    is_sev1_or_high = (
        risk in (ApprovalRiskLevel.CRITICAL, ApprovalRiskLevel.HIGH, ApprovalRiskLevel.MEDIUM)
        or severity.upper() in ("SEV1", "CRITICAL")
    )
    if is_sev1_or_high:
        status = ApprovalStatus.PENDING
    elif allow_auto_approve_low_risk and risk == ApprovalRiskLevel.LOW:
        status = ApprovalStatus.AUTO_APPROVED
    else:
        status = ApprovalStatus.PENDING

    return ApprovalRequest(
        request_id=request_id,
        issue_key=issue_key,
        severity=severity,
        subcategory=subcategory,
        proposed_action=proposed_action,
        risk_level=risk,
        reasoning=reasoning,
        status=status,
    )


def process_approval_decision(
    request: ApprovalRequest, decision: ApprovalStatus, approved_by: str = "engineer-on-call"
) -> ApprovalRequest:
    """Process explicit human approval or rejection of an approval request."""

    if decision not in (ApprovalStatus.APPROVED, ApprovalStatus.REJECTED):
        raise ValueError(f"Invalid approval decision {decision!r}; must be APPROVED or REJECTED")

    now = datetime.now(UTC)
    data = request.model_dump()
    data.update(
        {
            "status": decision,
            "approved_by": approved_by,
            "approved_at": now,
        }
    )
    return ApprovalRequest(**data)


__all__ = [
    "create_approval_request",
    "evaluate_remediation_risk",
    "process_approval_decision",
]
