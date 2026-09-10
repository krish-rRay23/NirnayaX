"""Tests for the agentic diagnosis state machine workflow."""

from __future__ import annotations

import pytest

from nirnayax.agent import (
    DiagnosisWorkflow,
    SimulatedEnvironment,
    WorkflowStatus,
)
from nirnayax.agent.types import Observation, WorkflowState
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


def test_workflow_remediation_verification_failure_escalates(
    trained_model: TriageModel,
    runbook_retriever: HybridRetriever,
    incident_retriever: HybridRetriever,
    sample_draft: TicketDraft,
    monkeypatch: pytest.MonkeyPatch,
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

    # Monkeypatch verify_service_recovery to simulate a failed recovery verification
    def mock_failed_verification(*args: object, **kwargs: object) -> Observation:
        return Observation(
            tool_name="recovery_verifier",
            target_service=service,
            metric_name="service_health_score",
            value=0.0,
            passed=False,
            details="Verification failed: metric error rate remained elevated",
        )

    import nirnayax.agent.workflow as wf_module
    monkeypatch.setattr(wf_module, "verify_service_recovery", mock_failed_verification)

    final_state = workflow.run(sample_draft, auto_approve_pending=True)

    assert final_state.status == WorkflowStatus.ESCALATED
    assert len(final_state.verification_observations) > 0
    assert not final_state.verification_observations[0].passed


def test_workflow_audit_trace_propagation(
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

    events = workflow.audit_logger.get_events()
    assert len(events) >= 3
    assert final_state.trace_id is not None

    for ev in events:
        assert ev.trace_id == final_state.trace_id
        assert ev.incident_id is not None
        assert len(ev.incident_id) > 0
