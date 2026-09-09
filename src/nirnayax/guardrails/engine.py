"""Central policy-driven Guardrail Engine for NirnayaX security and safety enforcement."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..ml import TicketDraft
from .injection import PromptInjectionDetector
from .policy import GuardrailPolicyConfig
from .sanitizer import PIISanitizer
from .types import (
    GuardrailCheckName,
    GuardrailCheckResult,
    GuardrailEvaluationResult,
)


class GuardrailEngine:
    """Centralized policy engine executing and enforcing security guardrails."""

    def __init__(self, policy: GuardrailPolicyConfig | None = None) -> None:
        self.policy = policy or GuardrailPolicyConfig()
        self.injection_detector = PromptInjectionDetector(self.policy)

    def validate_input(
        self, draft: TicketDraft, trace_id: str | None = None
    ) -> GuardrailEvaluationResult:
        """Validate input ticket draft for prompt injection, length limits, and sanitize PII.

        Returns GuardrailEvaluationResult with sanitized_draft if allowed.
        """
        tid = trace_id or f"TRC-{uuid4().hex[:8].upper()}"
        checks: list[GuardrailCheckResult] = []
        blocked_by: list[str] = []

        # 1. Length validation
        total_len = len(draft.title) + len(draft.description)
        if total_len > self.policy.max_input_length:
            msg = (
                f"Input text exceeds maximum length "
                f"({total_len} > {self.policy.max_input_length})."
            )
            checks.append(
                GuardrailCheckResult(
                    check_name=GuardrailCheckName.INPUT_VALIDATION,
                    passed=False,
                    reason=msg,
                )
            )
            blocked_by.append("Input length limit exceeded")
        else:
            checks.append(
                GuardrailCheckResult(
                    check_name=GuardrailCheckName.INPUT_VALIDATION,
                    passed=True,
                    reason="Input text length within limits.",
                )
            )

        # 2. Prompt injection detection
        inj_title, r_title = self.injection_detector.detect(draft.title)
        inj_desc, r_desc = self.injection_detector.detect(draft.description)

        if inj_title or inj_desc:
            reason = r_title if inj_title else r_desc
            checks.append(
                GuardrailCheckResult(
                    check_name=GuardrailCheckName.PROMPT_INJECTION,
                    passed=False,
                    reason=reason,
                )
            )
            blocked_by.append(reason)
        else:
            checks.append(
                GuardrailCheckResult(
                    check_name=GuardrailCheckName.PROMPT_INJECTION,
                    passed=True,
                    reason="No prompt injection detected.",
                )
            )

        # 3. PII & Sensitive data protection
        sanitized_draft, redaction_count = PIISanitizer.sanitize_draft(draft)
        checks.append(
            GuardrailCheckResult(
                check_name=GuardrailCheckName.PII_PROTECTION,
                passed=True,
                reason=f"PII protection active. Sanitized {redaction_count} sensitive fields.",
                sanitized_content=sanitized_draft.description,
                metadata={"redactions": redaction_count},
            )
        )

        allowed = len(blocked_by) == 0 if self.policy.fail_closed else True

        return GuardrailEvaluationResult(
            allowed=allowed,
            trace_id=tid,
            checks=checks,
            blocked_by=blocked_by,
            sanitized_draft=sanitized_draft,
        )

    def validate_gating(
        self,
        *,
        confidence: float,
        runbook_count: int,
        incident_count: int,
        decision: str,
    ) -> GuardrailCheckResult:
        """Validate confidence score and evidence requirements for decisions."""
        if not self.policy.enable_evidence_enforcement:
            return GuardrailCheckResult(
                check_name=GuardrailCheckName.EVIDENCE_CONFIDENCE,
                passed=True,
                reason="Evidence enforcement disabled by policy.",
            )

        if decision.upper() == "REMEDIATE":
            if confidence < self.policy.min_confidence_threshold:
                c_msg = (
                    f"Confidence score ({confidence:.1%}) below threshold "
                    f"({self.policy.min_confidence_threshold:.1%})."
                )
                return GuardrailCheckResult(
                    check_name=GuardrailCheckName.EVIDENCE_CONFIDENCE,
                    passed=False,
                    reason=c_msg,
                )

            if runbook_count < self.policy.min_runbook_evidence_count:
                e_msg = (
                    f"Insufficient runbook evidence matches "
                    f"({runbook_count} < {self.policy.min_runbook_evidence_count})."
                )
                return GuardrailCheckResult(
                    check_name=GuardrailCheckName.EVIDENCE_CONFIDENCE,
                    passed=False,
                    reason=e_msg,
                )

        return GuardrailCheckResult(
            check_name=GuardrailCheckName.EVIDENCE_CONFIDENCE,
            passed=True,
            reason="Confidence score and evidence count meet policy requirements.",
            metadata={
                "confidence": confidence,
                "runbook_count": runbook_count,
                "incident_count": incident_count,
            },
        )

    def validate_tool_call(
        self, tool_name: str, kwargs: dict[str, Any]
    ) -> GuardrailCheckResult:
        """Validate tool invocation against allowlists and inspect arguments for security."""
        if not self.policy.enable_tool_allowlisting:
            return GuardrailCheckResult(
                check_name=GuardrailCheckName.TOOL_ALLOWLIST,
                passed=True,
                reason="Tool allowlisting disabled by policy.",
            )

        if tool_name not in self.policy.allowed_tools:
            return GuardrailCheckResult(
                check_name=GuardrailCheckName.TOOL_ALLOWLIST,
                passed=False,
                reason=f"Tool '{tool_name}' is not in the allowed tools list.",
            )

        # Check remediation action if applicable
        if tool_name == "execute_remediation":
            action = kwargs.get("action", "")
            if action not in self.policy.allowed_remediation_actions:
                return GuardrailCheckResult(
                    check_name=GuardrailCheckName.TOOL_ALLOWLIST,
                    passed=False,
                    reason=f"Remediation action '{action}' is not in allowed remediation actions.",
                )

        # Check argument text for prompt injection / shell injection
        for k, v in kwargs.items():
            if isinstance(v, str):
                inj, r = self.injection_detector.detect(v)
                if inj:
                    return GuardrailCheckResult(
                        check_name=GuardrailCheckName.TOOL_ALLOWLIST,
                        passed=False,
                        reason=f"Dangerous payload in tool parameter '{k}': {r}",
                    )

        return GuardrailCheckResult(
            check_name=GuardrailCheckName.TOOL_ALLOWLIST,
            passed=True,
            reason=f"Tool '{tool_name}' invocation validated successfully.",
        )

    def validate_jira_action(
        self,
        *,
        action_type: str,
        issue_key: str,
        from_status: str,
        to_status: str,
        priority: str,
        human_approved: bool = False,
    ) -> GuardrailCheckResult:
        """Validate Jira transitions and actions against safety policies."""
        if not self.policy.enable_jira_guardrails:
            return GuardrailCheckResult(
                check_name=GuardrailCheckName.JIRA_SAFETY,
                passed=True,
                reason="Jira guardrails disabled by policy.",
            )

        if (
            action_type in ("delete_issue", "delete_comment", "overwrite_field")
            and self.policy.prohibit_destructive_jira_actions
        ):
            return GuardrailCheckResult(
                check_name=GuardrailCheckName.JIRA_SAFETY,
                passed=False,
                reason=f"Destructive Jira operation '{action_type}' prohibited by safety policy.",
            )

        # Prohibit auto-closing Critical / SEV1 / P1 incidents
        is_critical = priority.upper() in ("P1", "CRITICAL", "SEV1")
        is_closing = to_status.lower() in ("resolved", "closed", "done")

        if (
            is_critical
            and is_closing
            and not human_approved
            and self.policy.prohibit_auto_close_critical
        ):
            close_msg = (
                f"Auto-closing Critical/P1 Jira issue '{issue_key}' "
                f"without explicit human approval is prohibited."
            )
            return GuardrailCheckResult(
                check_name=GuardrailCheckName.JIRA_SAFETY,
                passed=False,
                reason=close_msg,
            )

        return GuardrailCheckResult(
            check_name=GuardrailCheckName.JIRA_SAFETY,
            passed=True,
            reason=f"Jira action '{action_type}' to state '{to_status}' permitted.",
        )
