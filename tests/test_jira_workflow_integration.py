"""Tests for workflow integration with Jira and human approval gate."""

from __future__ import annotations

from nirnayax.agent import DiagnosisWorkflow, SimulatedEnvironment, WorkflowStatus
from nirnayax.jira import JiraIssueStatus, MockJiraAdapter
from nirnayax.ml import TicketDraft, TriageModel
from nirnayax.retrieval import HybridRetriever


def test_jira_workflow_with_human_approval_resolution(
    trained_model: TriageModel,
    runbook_retriever: HybridRetriever,
    incident_retriever: HybridRetriever,
    sample_draft: TicketDraft,
) -> None:
    service = sample_draft.affected_service or "core-router"
    env = SimulatedEnvironment({service: "BGP_ROUTING"})
    jira_adapter = MockJiraAdapter()

    workflow = DiagnosisWorkflow(
        model=trained_model,
        runbook_retriever=runbook_retriever,
        incident_retriever=incident_retriever,
        env=env,
        jira_adapter=jira_adapter,
        min_confidence_threshold=0.50,
        auto_approve_low_risk=False,
    )

    # 1. Run until AWAITING_APPROVAL
    state = workflow.run(sample_draft, auto_approve_pending=False)
    assert state.status == WorkflowStatus.AWAITING_APPROVAL
    assert state.jira_issue_key is not None
    assert state.approval_request is not None

    jira_issue_pending = jira_adapter.get_issue(state.jira_issue_key or "")
    assert jira_issue_pending is not None
    assert jira_issue_pending.status == JiraIssueStatus.AWAITING_APPROVAL.value

    # 2. Grant explicit human approval
    state = workflow.approve(state, approved_by="senior-oncall")

    # 3. Resume workflow to completion
    final_state = workflow.run_state(state, auto_approve_pending=True)
    assert final_state.status == WorkflowStatus.RESOLVED

    jira_issue_done = jira_adapter.get_issue(final_state.jira_issue_key or "")
    assert jira_issue_done is not None
    assert jira_issue_done.status == JiraIssueStatus.RESOLVED.value
    assert len(jira_issue_done.comments) >= 5


def test_jira_workflow_with_rejection_escalation(
    trained_model: TriageModel,
    runbook_retriever: HybridRetriever,
    incident_retriever: HybridRetriever,
    sample_draft: TicketDraft,
) -> None:
    service = sample_draft.affected_service or "core-router"
    env = SimulatedEnvironment({service: "BGP_ROUTING"})
    jira_adapter = MockJiraAdapter()

    workflow = DiagnosisWorkflow(
        model=trained_model,
        runbook_retriever=runbook_retriever,
        incident_retriever=incident_retriever,
        env=env,
        jira_adapter=jira_adapter,
        min_confidence_threshold=0.50,
    )

    state = workflow.run(sample_draft, auto_approve_pending=False)
    assert state.status == WorkflowStatus.AWAITING_APPROVAL

    # Explicit rejection
    state = workflow.reject(state, approved_by="oncall-lead")
    final_state = workflow.step(state)  # AWAITING_APPROVAL (rejected) -> ESCALATION
    final_state = workflow.step(final_state)  # ESCALATION -> ESCALATED

    assert final_state.status == WorkflowStatus.ESCALATED
    jira_issue = jira_adapter.get_issue(final_state.jira_issue_key or "")
    assert jira_issue is not None
    assert jira_issue.status == JiraIssueStatus.ESCALATED.value
