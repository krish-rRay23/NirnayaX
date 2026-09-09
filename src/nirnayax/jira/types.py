"""Typed data contracts for Jira integration, action records, and human approval."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Immutable, strict base (mirrors domain / ML / retrieval / agent conventions)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ApprovalRiskLevel(StrEnum):
    """Risk classification for proposed remediation actions."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ApprovalStatus(StrEnum):
    """Status of a human approval request."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    AUTO_APPROVED = "AUTO_APPROVED"


class JiraIssueStatus(StrEnum):
    """Standard Jira issue statuses for NirnayaX incident lifecycle."""

    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    AWAITING_APPROVAL = "Awaiting Approval"
    RESOLVED = "Resolved"
    ESCALATED = "Escalated"


class JiraIssue(_Frozen):
    """A Jira issue tracking an incident ticket."""

    key: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    description: str = Field(min_length=1)
    issue_type: str = "Incident"
    status: str = "Open"
    priority: str = "P2"
    assignee: str | None = None
    comments: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    created_at: datetime
    updated_at: datetime


class JiraActionRecord(_Frozen):
    """Audit log record for an operation performed against Jira."""

    action_type: str
    issue_key: str
    timestamp: datetime
    details: str

    def render(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        return f"[{ts}] JIRA [{self.issue_key}] {self.action_type}: {self.details}"


class ApprovalRequest(_Frozen):
    """Human approval request record for risky or Critical remediation actions."""

    request_id: str
    issue_key: str
    severity: str
    subcategory: str
    proposed_action: str
    risk_level: ApprovalRiskLevel
    reasoning: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: str | None = None
    approved_at: datetime | None = None

    def render(self) -> str:
        approver = f" by {self.approved_by}" if self.approved_by else ""
        return (
            f"ApprovalRequest #{self.request_id} [{self.issue_key}] "
            f"risk={self.risk_level.value} status={self.status.value}{approver}: "
            f"Action '{self.proposed_action}' on {self.subcategory}"
        )


__all__ = [
    "ApprovalRequest",
    "ApprovalRiskLevel",
    "ApprovalStatus",
    "JiraActionRecord",
    "JiraIssue",
    "JiraIssueStatus",
]
