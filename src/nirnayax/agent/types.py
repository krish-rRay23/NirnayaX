"""Typed data contracts for the agentic diagnosis and decision workflow."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..jira.types import ApprovalRequest
from ..ml import TicketDraft, TriagePrediction
from ..retrieval import RetrievalResult


class _Frozen(BaseModel):
    """Immutable, strict base (mirrors domain / ML / retrieval conventions)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class WorkflowStatus(StrEnum):
    """Lifecycle states of the agentic diagnosis state machine."""

    NEW = "NEW"
    TRIAGED = "TRIAGED"
    CORRELATED = "CORRELATED"
    DIAGNOSING = "DIAGNOSING"
    DECISION = "DECISION"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    REMEDIATION = "REMEDIATION"
    ESCALATION = "ESCALATION"
    VERIFICATION = "VERIFICATION"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"


class Observation(_Frozen):
    """A diagnostic or verification observation from a simulated tool."""

    tool_name: str
    target_service: str
    metric_name: str
    value: float
    passed: bool
    details: str

    def render(self) -> str:
        status_flag = "PASS" if self.passed else "FAIL"
        hdr = f"[{self.tool_name}] {self.target_service}::{self.metric_name}={self.value:g}"
        return f"{hdr} ({status_flag}) - {self.details}"


class RemediationAction(_Frozen):
    """A remediation action executed by a simulated tool."""

    tool_name: str
    target_service: str
    action_taken: str
    success: bool
    details: str

    def render(self) -> str:
        flag = "SUCCESS" if self.success else "FAILED"
        hdr = f"[{self.tool_name}] {self.target_service} -> {self.action_taken}"
        return f"{hdr} ({flag}) - {self.details}"


class DecisionType(StrEnum):
    """Outcome decision at the gating stage."""

    REMEDIATE = "REMEDIATE"
    ESCALATE = "ESCALATE"


class DecisionOutcome(_Frozen):
    """The outcome of the confidence and evidence gate."""

    decision: DecisionType
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    suggested_action: str | None = None

    def render(self) -> str:
        action = f" -> {self.suggested_action}" if self.suggested_action else ""
        hdr = f"{self.decision.value} (confidence={self.confidence_score:.1%}){action}"
        return f"{hdr}: {self.reasoning}"


class WorkflowTransition(_Frozen):
    """Audit log entry for a state machine transition."""

    from_status: WorkflowStatus
    to_status: WorkflowStatus
    timestamp: datetime
    summary: str

    def render(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        return f"[{ts}] {self.from_status.value} -> {self.to_status.value}: {self.summary}"


class WorkflowState(_Frozen):
    """Immutable state container for an incident undergoing agentic diagnosis."""

    draft: TicketDraft
    status: WorkflowStatus = WorkflowStatus.NEW
    prediction: TriagePrediction | None = None
    retrieved_runbooks: tuple[RetrievalResult, ...] = ()
    similar_incidents: tuple[RetrievalResult, ...] = ()
    observations: tuple[Observation, ...] = ()
    remediation_action: RemediationAction | None = None
    verification_observations: tuple[Observation, ...] = ()
    decision: DecisionOutcome | None = None
    jira_issue_key: str | None = None
    approval_request: ApprovalRequest | None = None
    history: tuple[WorkflowTransition, ...] = ()

    def copy_with(self, **changes: Any) -> WorkflowState:
        """Create a updated immutable state copy with specified field changes."""

        data = self.model_dump(exclude_unset=True)
        data.update(changes)
        return WorkflowState(**data)


__all__ = [
    "DecisionOutcome",
    "DecisionType",
    "Observation",
    "RemediationAction",
    "WorkflowState",
    "WorkflowStatus",
    "WorkflowTransition",
]
