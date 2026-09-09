"""Typed state machine workflow for diagnosis, correlation, gating, and remediation."""

from __future__ import annotations

from datetime import UTC, datetime

from ..guardrails import AuditEventStatus, AuditLogger, GuardrailEngine
from ..jira import JiraAdapter, JiraIssueStatus
from ..jira.types import ApprovalStatus
from ..ml import TicketDraft, TriageModel
from ..retrieval import HybridRetriever
from .approval import create_approval_request, process_approval_decision
from .gating import MIN_CONFIDENCE_THRESHOLD, evaluate_confidence_gate
from .tools import (
    SimulatedEnvironment,
    check_service_metrics,
    execute_remediation,
    fetch_service_logs,
    verify_service_recovery,
)
from .types import (
    DecisionType,
    WorkflowState,
    WorkflowStatus,
    WorkflowTransition,
)


class DiagnosisWorkflow:
    """State machine graph executor for the agentic triage and diagnosis pipeline."""

    def __init__(
        self,
        *,
        model: TriageModel,
        runbook_retriever: HybridRetriever,
        incident_retriever: HybridRetriever,
        env: SimulatedEnvironment | None = None,
        jira_adapter: JiraAdapter | None = None,
        guardrail_engine: GuardrailEngine | None = None,
        audit_logger: AuditLogger | None = None,
        min_confidence_threshold: float = MIN_CONFIDENCE_THRESHOLD,
        auto_approve_low_risk: bool = False,
    ) -> None:
        self.model = model
        self.runbook_retriever = runbook_retriever
        self.incident_retriever = incident_retriever
        self.env = env or SimulatedEnvironment()
        self.jira_adapter = jira_adapter
        self.guardrail_engine = guardrail_engine or GuardrailEngine()
        self.audit_logger = audit_logger or AuditLogger()
        self.min_confidence_threshold = min_confidence_threshold
        self.auto_approve_low_risk = auto_approve_low_risk

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def approve(
        self, state: WorkflowState, approved_by: str = "engineer-on-call"
    ) -> WorkflowState:
        """Explicitly approve a pending remediation approval request."""

        if state.status != WorkflowStatus.AWAITING_APPROVAL or state.approval_request is None:
            return state
        updated_req = process_approval_decision(
            state.approval_request, ApprovalStatus.APPROVED, approved_by
        )
        summary = f"Explicit human approval granted by {approved_by}."
        transition = WorkflowTransition(
            from_status=state.status,
            to_status=WorkflowStatus.AWAITING_APPROVAL,
            timestamp=self._now(),
            summary=summary,
        )
        return state.copy_with(
            approval_request=updated_req,
            history=(*state.history, transition),
        )

    def reject(
        self, state: WorkflowState, approved_by: str = "engineer-on-call"
    ) -> WorkflowState:
        """Explicitly reject a pending remediation approval request."""

        if state.status != WorkflowStatus.AWAITING_APPROVAL or state.approval_request is None:
            return state
        updated_req = process_approval_decision(
            state.approval_request, ApprovalStatus.REJECTED, approved_by
        )
        summary = f"Human approval rejected by {approved_by}."
        transition = WorkflowTransition(
            from_status=state.status,
            to_status=WorkflowStatus.AWAITING_APPROVAL,
            timestamp=self._now(),
            summary=summary,
        )
        return state.copy_with(
            approval_request=updated_req,
            history=(*state.history, transition),
        )

    def step(self, state: WorkflowState) -> WorkflowState:
        """Execute a single step transition in the state machine."""

        if state.status == WorkflowStatus.NEW:
            return self._step_new(state)
        elif state.status == WorkflowStatus.TRIAGED:
            return self._step_triaged(state)
        elif state.status == WorkflowStatus.CORRELATED:
            return self._step_correlated(state)
        elif state.status == WorkflowStatus.DIAGNOSING:
            return self._step_diagnosing(state)
        elif state.status == WorkflowStatus.DECISION:
            return self._step_decision(state)
        elif state.status == WorkflowStatus.AWAITING_APPROVAL:
            return self._step_awaiting_approval(state)
        elif state.status == WorkflowStatus.REMEDIATION:
            return self._step_remediation(state)
        elif state.status == WorkflowStatus.VERIFICATION:
            return self._step_verification(state)
        elif state.status == WorkflowStatus.ESCALATION:
            return self._step_escalation(state)
        elif state.status in (WorkflowStatus.RESOLVED, WorkflowStatus.ESCALATED):
            return state
        return state

    def _step_new(self, state: WorkflowState) -> WorkflowState:
        # Validate input via GuardrailEngine
        eval_result = self.guardrail_engine.validate_input(state.draft)

        if not eval_result.allowed:
            self.audit_logger.log_event(
                trace_id=eval_result.trace_id,
                incident_id=state.jira_issue_key or "N/A",
                actor="GuardrailEngine",
                decision="BLOCKED",
                confidence=0.0,
                tool_action="input_validation",
                guardrail_results=[c.to_dict() for c in eval_result.checks],
                status=AuditEventStatus.BLOCKED,
            )

            next_status = WorkflowStatus.ESCALATED
            summary = f"Guardrail violation blocked execution: {'; '.join(eval_result.blocked_by)}"
            jira_key = state.jira_issue_key

            if self.jira_adapter is not None:
                if jira_key is None:
                    issue = self.jira_adapter.create_issue(
                        summary=state.draft.title,
                        description=state.draft.description,
                        priority="P1",
                        labels=("nirnayax", "guardrail-blocked"),
                    )
                    jira_key = issue.key

                self.jira_adapter.add_comment(
                    jira_key, f"GUARDRAIL BLOCKED: {summary}"
                )
                self.jira_adapter.transition_issue(jira_key, JiraIssueStatus.ESCALATED.value)

            transition = WorkflowTransition(
                from_status=state.status,
                to_status=next_status,
                timestamp=self._now(),
                summary=summary,
            )
            return state.copy_with(
                status=next_status,
                jira_issue_key=jira_key,
                history=(*state.history, transition),
            )

        sanitized_draft = eval_result.sanitized_draft or state.draft
        prediction = self.model.predict(sanitized_draft)
        jira_key = state.jira_issue_key

        if self.jira_adapter is not None:
            if jira_key is None:
                issue = self.jira_adapter.create_issue(
                    summary=sanitized_draft.title,
                    description=sanitized_draft.description,
                    priority=prediction.priority.label,
                    labels=("nirnayax", "auto-triage"),
                )
                jira_key = issue.key

            self.jira_adapter.add_comment(
                jira_key,
                f"ML Triage Prediction:\n{prediction.render()}",
            )
            self.jira_adapter.transition_issue(jira_key, JiraIssueStatus.IN_PROGRESS.value)

        next_status = WorkflowStatus.TRIAGED
        summary = (
            f"ML triage predicted category={prediction.category.label}, "
            f"subcategory={prediction.subcategory.label} ({prediction.subcategory.confidence:.1%})"
        )
        transition = WorkflowTransition(
            from_status=state.status,
            to_status=next_status,
            timestamp=self._now(),
            summary=summary,
        )
        return state.copy_with(
            status=next_status,
            draft=sanitized_draft,
            prediction=prediction,
            jira_issue_key=jira_key,
            history=(*state.history, transition),
        )

    def _step_triaged(self, state: WorkflowState) -> WorkflowState:
        query = f"{state.draft.title}\n{state.draft.description}"
        filters = {}
        if state.prediction:
            filters["category"] = state.prediction.category.label

        runbooks = tuple(self.runbook_retriever.retrieve(query, k=3, filters=filters or None))
        incidents = tuple(self.incident_retriever.retrieve(query, k=5))

        if self.jira_adapter is not None and state.jira_issue_key:
            msg = (
                f"RAG Evidence Correlation:\n"
                f"- Runbooks retrieved: {len(runbooks)}\n"
                f"- Similar incidents correlated: {len(incidents)}"
            )
            self.jira_adapter.add_comment(state.jira_issue_key, msg)

        next_status = WorkflowStatus.CORRELATED
        summary = (
            f"Retrieved {len(runbooks)} matching runbooks and {len(incidents)} similar incidents."
        )
        transition = WorkflowTransition(
            from_status=state.status,
            to_status=next_status,
            timestamp=self._now(),
            summary=summary,
        )
        return state.copy_with(
            status=next_status,
            retrieved_runbooks=runbooks,
            similar_incidents=incidents,
            history=(*state.history, transition),
        )

    def _step_correlated(self, state: WorkflowState) -> WorkflowState:
        service = state.draft.affected_service or "default-service"
        subcat_hint = state.prediction.subcategory.label if state.prediction else None

        obs1 = check_service_metrics(self.env, service, subcat_hint)
        obs2 = fetch_service_logs(self.env, service)
        observations = (obs1, obs2)

        if self.jira_adapter is not None and state.jira_issue_key:
            msg = "Diagnostic Observations:\n" + "\n".join(o.render() for o in observations)
            self.jira_adapter.add_comment(state.jira_issue_key, msg)

        next_status = WorkflowStatus.DIAGNOSING
        summary = f"Observed {len(observations)} diagnostic metrics/logs for service {service}."
        transition = WorkflowTransition(
            from_status=state.status,
            to_status=next_status,
            timestamp=self._now(),
            summary=summary,
        )
        return state.copy_with(
            status=next_status,
            observations=observations,
            history=(*state.history, transition),
        )

    def _step_diagnosing(self, state: WorkflowState) -> WorkflowState:
        outcome = evaluate_confidence_gate(
            prediction=state.prediction,
            retrieved_runbooks=state.retrieved_runbooks,
            similar_incidents=state.similar_incidents,
            observations=state.observations,
            min_threshold=self.min_confidence_threshold,
        )

        if self.jira_adapter is not None and state.jira_issue_key:
            self.jira_adapter.add_comment(
                state.jira_issue_key, f"Gating Evaluation:\n{outcome.render()}"
            )

        next_status = WorkflowStatus.DECISION
        summary = (
            f"Confidence gate evaluated: {outcome.decision.value} "
            f"(score={outcome.confidence_score:.1%})."
        )
        transition = WorkflowTransition(
            from_status=state.status,
            to_status=next_status,
            timestamp=self._now(),
            summary=summary,
        )
        return state.copy_with(
            status=next_status,
            decision=outcome,
            history=(*state.history, transition),
        )

    def _step_decision(self, state: WorkflowState) -> WorkflowState:
        if state.decision and state.decision.decision == DecisionType.REMEDIATE:
            proposed_action = state.decision.suggested_action or "execute_remediation"
            subcat = state.prediction.subcategory.label if state.prediction else "UNKNOWN"
            sev = state.prediction.priority.label if state.prediction else "SEV2"

            app_req = create_approval_request(
                request_id=f"REQ-{len(state.history) + 1}",
                issue_key=state.jira_issue_key or "N/A",
                severity=sev,
                subcategory=subcat,
                proposed_action=proposed_action,
                reasoning=state.decision.reasoning,
                allow_auto_approve_low_risk=self.auto_approve_low_risk,
            )

            if app_req.status == ApprovalStatus.AUTO_APPROVED:
                next_status = WorkflowStatus.REMEDIATION
                summary = "Confidence gate passed; low risk auto-approved for remediation."
            else:
                next_status = WorkflowStatus.AWAITING_APPROVAL
                summary = f"Awaiting human approval ({app_req.risk_level.value} risk)."
                if self.jira_adapter is not None and state.jira_issue_key:
                    self.jira_adapter.add_comment(
                        state.jira_issue_key,
                        f"Human Approval Required:\n{app_req.render()}",
                    )
                    self.jira_adapter.transition_issue(
                        state.jira_issue_key, JiraIssueStatus.AWAITING_APPROVAL.value
                    )

            transition = WorkflowTransition(
                from_status=state.status,
                to_status=next_status,
                timestamp=self._now(),
                summary=summary,
            )
            return state.copy_with(
                status=next_status,
                approval_request=app_req,
                history=(*state.history, transition),
            )

        next_status = WorkflowStatus.ESCALATION
        reason = state.decision.reasoning if state.decision else "Insufficient evidence"
        summary = f"Confidence gate failed ({reason}); escalating ticket."
        if self.jira_adapter is not None and state.jira_issue_key:
            self.jira_adapter.transition_issue(
                state.jira_issue_key, JiraIssueStatus.ESCALATED.value
            )

        transition = WorkflowTransition(
            from_status=state.status,
            to_status=next_status,
            timestamp=self._now(),
            summary=summary,
        )
        return state.copy_with(
            status=next_status,
            history=(*state.history, transition),
        )

    def _step_awaiting_approval(self, state: WorkflowState) -> WorkflowState:
        req = state.approval_request
        if req is None:
            return self._step_escalation(state)

        if req.status == ApprovalStatus.APPROVED:
            next_status = WorkflowStatus.REMEDIATION
            summary = f"Approval granted by {req.approved_by}. Proceeding to remediation."
            if self.jira_adapter is not None and state.jira_issue_key:
                self.jira_adapter.add_comment(
                    state.jira_issue_key,
                    f"Human Approval Granted by {req.approved_by}.",
                )
                self.jira_adapter.transition_issue(
                    state.jira_issue_key, JiraIssueStatus.IN_PROGRESS.value
                )

            transition = WorkflowTransition(
                from_status=state.status,
                to_status=next_status,
                timestamp=self._now(),
                summary=summary,
            )
            return state.copy_with(
                status=next_status,
                history=(*state.history, transition),
            )
        elif req.status == ApprovalStatus.REJECTED:
            next_status = WorkflowStatus.ESCALATION
            summary = f"Approval rejected by {req.approved_by}. Escalating ticket."
            if self.jira_adapter is not None and state.jira_issue_key:
                self.jira_adapter.add_comment(
                    state.jira_issue_key,
                    f"Human Approval Rejected by {req.approved_by}.",
                )
                self.jira_adapter.transition_issue(
                    state.jira_issue_key, JiraIssueStatus.ESCALATED.value
                )

            transition = WorkflowTransition(
                from_status=state.status,
                to_status=next_status,
                timestamp=self._now(),
                summary=summary,
            )
            return state.copy_with(
                status=next_status,
                history=(*state.history, transition),
            )

        # Still pending approval: stay in AWAITING_APPROVAL
        return state

    def _step_remediation(self, state: WorkflowState) -> WorkflowState:
        service = state.draft.affected_service or "default-service"
        subcat = state.prediction.subcategory.label if state.prediction else "UNKNOWN"

        tool_check = self.guardrail_engine.validate_tool_call(
            "execute_remediation", {"action": subcat, "service": service}
        )
        if not tool_check.passed:
            self.audit_logger.log_event(
                trace_id="TRC-REMEDIATION-FAIL",
                incident_id=state.jira_issue_key or "N/A",
                actor="GuardrailEngine",
                decision="BLOCKED",
                confidence=0.0,
                tool_action=f"execute_remediation:{subcat}",
                guardrail_results=[tool_check.to_dict()],
                status=AuditEventStatus.BLOCKED,
            )
            next_status = WorkflowStatus.ESCALATED
            summary = f"Guardrail blocked tool execution: {tool_check.reason}"
            if self.jira_adapter is not None and state.jira_issue_key:
                self.jira_adapter.add_comment(
                    state.jira_issue_key, f"GUARDRAIL BLOCKED: {summary}"
                )
                self.jira_adapter.transition_issue(
                    state.jira_issue_key, JiraIssueStatus.ESCALATED.value
                )
            transition = WorkflowTransition(
                from_status=state.status,
                to_status=next_status,
                timestamp=self._now(),
                summary=summary,
            )
            return state.copy_with(
                status=next_status,
                history=(*state.history, transition),
            )

        action = execute_remediation(self.env, service, subcat)

        if self.jira_adapter is not None and state.jira_issue_key:
            self.jira_adapter.add_comment(
                state.jira_issue_key,
                f"Remediation Executed:\n{action.render()}",
            )

        next_status = WorkflowStatus.VERIFICATION
        summary = f"Executed remediation: {action.action_taken} ({action.details})."
        transition = WorkflowTransition(
            from_status=state.status,
            to_status=next_status,
            timestamp=self._now(),
            summary=summary,
        )
        return state.copy_with(
            status=next_status,
            remediation_action=action,
            history=(*state.history, transition),
        )

    def _step_verification(self, state: WorkflowState) -> WorkflowState:
        service = state.draft.affected_service or "default-service"
        subcat = state.prediction.subcategory.label if state.prediction else "UNKNOWN"

        verify_obs = verify_service_recovery(self.env, service, subcat)
        if verify_obs.passed:
            prio = state.prediction.priority.label if state.prediction else "SEV2"
            appr_statuses = (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED)
            human_appr = (
                state.approval_request is not None
                and state.approval_request.status in appr_statuses
            )
            jira_check = self.guardrail_engine.validate_jira_action(
                action_type="transition_issue",
                issue_key=state.jira_issue_key or "N/A",
                from_status=JiraIssueStatus.IN_PROGRESS.value,
                to_status=JiraIssueStatus.RESOLVED.value,
                priority=prio,
                human_approved=human_appr,
            )

            if not jira_check.passed:
                self.audit_logger.log_event(
                    trace_id="TRC-JIRA-SAFE-FAIL",
                    incident_id=state.jira_issue_key or "N/A",
                    actor="GuardrailEngine",
                    decision="BLOCKED",
                    confidence=0.0,
                    tool_action="transition_issue:RESOLVED",
                    guardrail_results=[jira_check.to_dict()],
                    status=AuditEventStatus.BLOCKED,
                )
                next_status = WorkflowStatus.ESCALATED
                summary = f"Jira guardrail blocked resolution: {jira_check.reason}"
                if self.jira_adapter is not None and state.jira_issue_key:
                    self.jira_adapter.add_comment(
                        state.jira_issue_key, f"GUARDRAIL BLOCKED: {summary}"
                    )
                    self.jira_adapter.transition_issue(
                        state.jira_issue_key, JiraIssueStatus.ESCALATED.value
                    )
                transition = WorkflowTransition(
                    from_status=state.status,
                    to_status=next_status,
                    timestamp=self._now(),
                    summary=summary,
                )
                return state.copy_with(
                    status=next_status,
                    history=(*state.history, transition),
                )

            next_status = WorkflowStatus.RESOLVED
            summary = f"Verification passed on {service}. Incident resolved."
            if self.jira_adapter is not None and state.jira_issue_key:
                self.jira_adapter.add_comment(
                    state.jira_issue_key,
                    f"Recovery Verification Passed: {verify_obs.details}",
                )
                self.jira_adapter.transition_issue(
                    state.jira_issue_key, JiraIssueStatus.RESOLVED.value
                )
        else:
            next_status = WorkflowStatus.ESCALATED
            summary = f"Verification failed on {service}. Escalating incident to L2."
            if self.jira_adapter is not None and state.jira_issue_key:
                self.jira_adapter.add_comment(
                    state.jira_issue_key,
                    f"Recovery Verification Failed: {verify_obs.details}",
                )
                self.jira_adapter.transition_issue(
                    state.jira_issue_key, JiraIssueStatus.ESCALATED.value
                )

        transition = WorkflowTransition(
            from_status=state.status,
            to_status=next_status,
            timestamp=self._now(),
            summary=summary,
        )
        return state.copy_with(
            status=next_status,
            verification_observations=(verify_obs,),
            history=(*state.history, transition),
        )

    def _step_escalation(self, state: WorkflowState) -> WorkflowState:
        next_status = WorkflowStatus.ESCALATED
        summary = "Ticket officially escalated to on-call engineering team."

        if self.jira_adapter is not None and state.jira_issue_key:
            self.jira_adapter.transition_issue(
                state.jira_issue_key, JiraIssueStatus.ESCALATED.value
            )

        transition = WorkflowTransition(
            from_status=state.status,
            to_status=next_status,
            timestamp=self._now(),
            summary=summary,
        )
        return state.copy_with(
            status=next_status,
            history=(*state.history, transition),
        )

    def run_state(
        self, state: WorkflowState, auto_approve_pending: bool = False
    ) -> WorkflowState:
        """Step an existing WorkflowState to a terminal state or pending approval."""

        terminal_states = {WorkflowStatus.RESOLVED, WorkflowStatus.ESCALATED}
        while state.status not in terminal_states:
            if state.status == WorkflowStatus.AWAITING_APPROVAL:
                req = state.approval_request
                if auto_approve_pending and req and req.status == ApprovalStatus.PENDING:
                    state = self.approve(state)
                elif req and req.status in (
                    ApprovalStatus.APPROVED,
                    ApprovalStatus.REJECTED,
                    ApprovalStatus.AUTO_APPROVED,
                ):
                    pass
                else:
                    break
            prev_status = state.status
            state = self.step(state)
            if state.status == prev_status and state.status == WorkflowStatus.AWAITING_APPROVAL:
                break
        return state

    def run(
        self, draft: TicketDraft, auto_approve_pending: bool = False
    ) -> WorkflowState:
        """Run the state machine from NEW for a draft to terminal state or pending approval."""

        return self.run_state(WorkflowState(draft=draft), auto_approve_pending=auto_approve_pending)


__all__ = ["DiagnosisWorkflow"]
