"""Typed data contracts and structures for security guardrails and audit logging."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..ml import TicketDraft


class GuardrailCheckName(str, Enum):  # noqa: UP042
    """Supported guardrail check types."""

    INPUT_VALIDATION = "INPUT_VALIDATION"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    PII_PROTECTION = "PII_PROTECTION"
    EVIDENCE_CONFIDENCE = "EVIDENCE_CONFIDENCE"
    TOOL_ALLOWLIST = "TOOL_ALLOWLIST"
    APPROVAL_RISK = "APPROVAL_RISK"
    JIRA_SAFETY = "JIRA_SAFETY"


class GuardrailCheckResult(BaseModel):
    """Result of a single guardrail check."""

    check_name: GuardrailCheckName
    passed: bool
    reason: str
    sanitized_content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert check result to dictionary for audit logging."""
        return {
            "check_name": self.check_name.value,
            "passed": self.passed,
            "reason": self.reason,
            "sanitized_content": self.sanitized_content,
            "metadata": self.metadata,
        }


class GuardrailEvaluationResult(BaseModel):
    """Overall result of a suite of guardrail checks."""

    allowed: bool
    trace_id: str = Field(default_factory=lambda: f"TRC-{uuid4().hex[:8].upper()}")
    checks: list[GuardrailCheckResult] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    sanitized_draft: TicketDraft | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AuditEventStatus(str, Enum):  # noqa: UP042
    """Status of an audited operation."""

    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"
    ESCALATED = "ESCALATED"
    PASSED = "PASSED"
    FAILED = "FAILED"


class AuditEvent(BaseModel):
    """Immutable audit record logging an operational or policy event."""

    trace_id: str
    incident_id: str
    actor: str
    decision: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    tool_action: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    guardrail_results: list[dict[str, Any]] = Field(default_factory=list)
    status: AuditEventStatus = AuditEventStatus.ALLOWED
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        """Render human-readable summary line for audit logging."""
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        evidence_str = f"[{len(self.evidence)} items]" if self.evidence else "none"
        return (
            f"[{ts}] AUDIT trace={self.trace_id} incident={self.incident_id} "
            f"actor='{self.actor}' status={self.status.value} decision='{self.decision}' "
            f"confidence={self.confidence:.1%} tool_action='{self.tool_action}' "
            f"evidence={evidence_str}"
        )
