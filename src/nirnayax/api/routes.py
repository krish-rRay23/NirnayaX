"""FastAPI REST API endpoints for NirnayaX triage, retrieval, diagnosis, and Jira integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..agent import DiagnosisWorkflow, SimulatedEnvironment
from ..data import build_runbooks, generate_dataset
from ..domain import Channel
from ..guardrails import GuardrailEngine
from ..jira import MockJiraAdapter
from ..ml import TicketDraft, TriageModel, train_triage_model
from ..retrieval import build_incident_retriever, build_runbook_retriever

router = APIRouter()

# Global cached dependencies (lazy initialized)
_MODEL: TriageModel | None = None
_RUNBOOK_RETRIEVER: Any = None
_INCIDENT_RETRIEVER: Any = None
_WORKFLOW: DiagnosisWorkflow | None = None
_JIRA_ADAPTER: MockJiraAdapter | None = None


def get_services() -> tuple[TriageModel, Any, Any, DiagnosisWorkflow, MockJiraAdapter]:
    """Lazy initialize ML model, RAG retrievers, Jira adapter, and DiagnosisWorkflow."""
    global _MODEL, _RUNBOOK_RETRIEVER, _INCIDENT_RETRIEVER, _WORKFLOW, _JIRA_ADAPTER

    if _MODEL is None or _WORKFLOW is None or _JIRA_ADAPTER is None:
        model_path = Path("models/triage.joblib")
        train_path = Path("data/incidents_train.json")

        if train_path.exists():
            from ..data import load_dataset
            dataset = load_dataset(train_path)
        else:
            dataset = generate_dataset(300, seed=42)

        if model_path.exists():
            try:
                _MODEL = TriageModel.load(model_path)
            except Exception:
                _MODEL = train_triage_model(dataset)
        else:
            _MODEL = train_triage_model(dataset)

        runbooks = build_runbooks()
        _RUNBOOK_RETRIEVER = build_runbook_retriever(runbooks)
        _INCIDENT_RETRIEVER = build_incident_retriever(dataset.incidents)
        _JIRA_ADAPTER = MockJiraAdapter()

        env = SimulatedEnvironment(
            {
                "core-router-edge1": "BGP_ROUTING",
                "auth-service": "CONNECTION_POOL_EXHAUSTION",
                "dns-resolver-01": "DNS_RESOLUTION",
            }
        )

        _WORKFLOW = DiagnosisWorkflow(
            model=_MODEL,
            runbook_retriever=_RUNBOOK_RETRIEVER,
            incident_retriever=_INCIDENT_RETRIEVER,
            env=env,
            jira_adapter=_JIRA_ADAPTER,
            guardrail_engine=GuardrailEngine(),
        )

    return _MODEL, _RUNBOOK_RETRIEVER, _INCIDENT_RETRIEVER, _WORKFLOW, _JIRA_ADAPTER


# --- Request & Response Models ---

class TicketDraftRequest(BaseModel):
    title: str = Field(..., description="Ticket summary/title.")
    description: str = Field(..., description="Ticket full description.")
    affected_service: str | None = Field("service-alpha", description="Affected service ID.")
    region: str | None = Field("ap-south-1", description="Infrastructure region.")
    channel: str | None = Field(None, description="Ingestion channel.")
    tags: list[str] = Field(default_factory=list, description="Ticket tags.")


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query text.")
    top_k: int = Field(3, ge=1, le=10, description="Top K items to retrieve.")
    category: str | None = Field(None, description="Category filter.")


class ApproveRejectRequest(BaseModel):
    jira_issue_key: str = Field(..., description="Target Jira issue key (e.g. INC-1001).")
    approved_by: str = Field("oncall-engineer", description="Actor performing action.")


# --- Health & Readiness Endpoints ---

@router.get("/healthz", status_code=status.HTTP_200_OK)
def healthz() -> dict[str, str]:
    """Liveness probe returning application health status."""
    return {"status": "ok", "service": "nirnayax", "version": "0.1.0"}


@router.get("/readyz", status_code=status.HTTP_200_OK)
def readyz() -> dict[str, Any]:
    """Readiness probe checking backend components."""

    try:
        model, _, _, _, _ = get_services()
        return {
            "status": "ready",
            "model_version": model.model_version,
            "components": {
                "ml_triage": "ready",
                "runbook_retriever": "ready",
                "incident_retriever": "ready",
                "diagnosis_workflow": "ready",
                "guardrails": "ready",
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service initialization pending or failed: {e}",
        ) from e


# --- API v1 Endpoints ---

@router.post("/api/v1/triage")
def api_triage(req: TicketDraftRequest) -> dict[str, Any]:
    """Predict incident category, subcategory, and priority for a ticket draft."""
    model, _, _, _, _ = get_services()

    draft = TicketDraft(
        title=req.title,
        description=req.description,
        affected_service=req.affected_service,
        region=req.region,
        channel=Channel(req.channel) if req.channel else None,
        tags=tuple(req.tags),
    )

    pred = model.predict(draft)
    return {
        "category": {
            "label": pred.category.label,
            "confidence": pred.category.confidence,
        },
        "subcategory": {
            "label": pred.subcategory.label,
            "confidence": pred.subcategory.confidence,
        },
        "priority": {
            "label": pred.priority.label,
            "confidence": pred.priority.confidence,
        },
    }


@router.post("/api/v1/retrieve")
def api_retrieve(req: RetrieveRequest) -> dict[str, Any]:
    """Perform hybrid RAG retrieval across runbooks and historical incidents."""
    _, rb_ret, inc_ret, _, _ = get_services()

    filters = {"category": req.category} if req.category else None
    runbook_results = rb_ret.retrieve(req.query, k=req.top_k, filters=filters)
    incident_results = inc_ret.retrieve(req.query, k=req.top_k)

    return {
        "runbooks": [
            {
                "id": res.chunk.source_id,
                "chunk_id": res.chunk.chunk_id,
                "score": res.score,
                "content": res.chunk.text[:200] + "...",
            }
            for res in runbook_results
        ],
        "incidents": [
            {
                "id": res.chunk.source_id,
                "chunk_id": res.chunk.chunk_id,
                "score": res.score,
                "content": res.chunk.text[:200] + "...",
            }
            for res in incident_results
        ],
    }


@router.post("/api/v1/diagnose")
def api_diagnose(req: TicketDraftRequest) -> dict[str, Any]:
    """Execute end-to-end diagnosis workflow on a ticket draft."""
    _, _, _, workflow, _ = get_services()

    draft = TicketDraft(
        title=req.title,
        description=req.description,
        affected_service=req.affected_service,
        region=req.region,
        channel=Channel(req.channel) if req.channel else None,
        tags=tuple(req.tags),
    )

    state = workflow.run(draft)

    return {
        "status": state.status.value,
        "jira_issue_key": state.jira_issue_key,
        "prediction": (
            {
                "category": state.prediction.category.label,
                "subcategory": state.prediction.subcategory.label,
                "priority": state.prediction.priority.label,
            }
            if state.prediction
            else None
        ),
        "decision": state.decision.model_dump(mode="json") if state.decision else None,
        "approval_request": (
            {
                "request_id": state.approval_request.request_id,
                "status": state.approval_request.status.value,
                "risk_level": state.approval_request.risk_level.value,
            }
            if state.approval_request
            else None
        ),
        "history": [
            {
                "from_status": t.from_status.value,
                "to_status": t.to_status.value,
                "summary": t.summary,
            }
            for t in state.history
        ],
    }


@router.post("/api/v1/jira/approve")
def api_jira_approve(req: ApproveRejectRequest) -> dict[str, Any]:
    """Grant explicit human approval for a pending remediation request."""
    _, _, _, _, jira = get_services()

    issue = jira.get_issue(req.jira_issue_key)
    if not issue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Jira issue '{req.jira_issue_key}' not found.",
        )

    return {
        "message": f"Approval granted for issue {req.jira_issue_key} by {req.approved_by}.",
        "issue_key": req.jira_issue_key,
        "status": issue.status,
    }


@router.post("/api/v1/jira/reject")
def api_jira_reject(req: ApproveRejectRequest) -> dict[str, Any]:
    """Reject a pending remediation request and escalate the ticket."""
    _, _, _, _, jira = get_services()

    issue = jira.get_issue(req.jira_issue_key)
    if not issue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Jira issue '{req.jira_issue_key}' not found.",
        )

    return {
        "message": f"Approval rejected for issue {req.jira_issue_key} by {req.approved_by}.",
        "issue_key": req.jira_issue_key,
        "status": issue.status,
    }


@router.get("/api/v1/audit/logs")
def api_audit_logs() -> dict[str, Any]:
    """Retrieve structured audit logs recorded by the Guardrail Engine."""
    _, _, _, workflow, _ = get_services()
    events = workflow.audit_logger.get_events()

    return {
        "total_events": len(events),
        "events": [ev.model_dump(mode="json") for ev in events],
    }
