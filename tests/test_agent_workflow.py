"""Tests for the agentic diagnosis state machine workflow."""

from __future__ import annotations

from nirnayax.agent import (
    DiagnosisWorkflow,
    SimulatedEnvironment,
    WorkflowStatus,
)
from nirnayax.ml import TicketDraft, TriageModel
from nirnayax.retrieval import HybridRetriever


def test_workflow_end_to_end_resolution(
    trained_model: TriageModel,
    runbook_retriever: HybridRetriever,
    incident_retriever: HybridRetriever,
    sample_draft: TicketDraft,
) -> None:
    service = sample_draft.affected_service or "core-router"
    env = SimulatedEnvironment({service: "BGP_ROUTING"})
    workflow = DiagnosisWorkflow(
        model=trained_model,
        runbook_retriever=runbook_retriever,
        incident_retriever=incident_retriever,
        env=env,
        min_confidence_threshold=0.50,
    )

    final_state = workflow.run(sample_draft, auto_approve_pending=True)

    assert final_state.status == WorkflowStatus.RESOLVED
    assert final_state.prediction is not None
    assert len(final_state.retrieved_runbooks) > 0
    assert len(final_state.observations) > 0
    assert final_state.decision is not None
    assert final_state.remediation_action is not None
    assert final_state.remediation_action.success
    assert len(final_state.history) >= 6


def test_workflow_escalation_on_high_confidence_threshold(
    trained_model: TriageModel,
    runbook_retriever: HybridRetriever,
    incident_retriever: HybridRetriever,
    sample_draft: TicketDraft,
) -> None:
    service = sample_draft.affected_service or "core-router"
    env = SimulatedEnvironment({service: "BGP_ROUTING"})
    workflow = DiagnosisWorkflow(
        model=trained_model,
        runbook_retriever=runbook_retriever,
        incident_retriever=incident_retriever,
        env=env,
        min_confidence_threshold=0.99,  # Unattainable threshold forces escalation
    )

    final_state = workflow.run(sample_draft)

    assert final_state.status == WorkflowStatus.ESCALATED
    assert final_state.decision is not None
    assert final_state.remediation_action is None


def test_workflow_step_by_step_transitions(
    trained_model: TriageModel,
    runbook_retriever: HybridRetriever,
    incident_retriever: HybridRetriever,
    sample_draft: TicketDraft,
) -> None:
    from nirnayax.agent.types import WorkflowState

    workflow = DiagnosisWorkflow(
        model=trained_model,
        runbook_retriever=runbook_retriever,
        incident_retriever=incident_retriever,
        min_confidence_threshold=0.50,
    )

    state = WorkflowState(draft=sample_draft)
    assert state.status == WorkflowStatus.NEW

    state = workflow.step(state)
    assert state.status == WorkflowStatus.TRIAGED

    state = workflow.step(state)
    assert state.status == WorkflowStatus.CORRELATED

    state = workflow.step(state)
    assert state.status == WorkflowStatus.DIAGNOSING

    state = workflow.step(state)
    assert state.status == WorkflowStatus.DECISION
