"""End-to-End evaluation suite measuring ML, RAG, gating, guardrails, and latency."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..agent import DiagnosisWorkflow, SimulatedEnvironment, WorkflowStatus
from ..data import build_runbooks
from ..domain import IncidentDataset
from ..guardrails import GuardrailEngine
from ..jira import MockJiraAdapter
from ..ml import TicketDraft, TriageModel
from ..retrieval import build_incident_retriever, build_runbook_retriever


@dataclass
class E2EEvaluationReport:
    """Quantitative report summarizing end-to-end performance metrics."""

    num_incidents: int
    ml_category_acc: float
    ml_subcategory_acc: float
    ml_priority_acc: float
    retrieval_recall_at_1: float
    retrieval_recall_at_3: float
    retrieval_mrr: float
    resolved_count: int
    awaiting_approval_count: int
    escalated_count: int
    auto_approved_count: int
    triage_mean_ms: float
    triage_p95_ms: float
    retrieval_mean_ms: float
    retrieval_p95_ms: float
    workflow_mean_ms: float
    workflow_p95_ms: float

    def render(self) -> str:
        """Render markdown-formatted summary report."""
        res_pct = self.resolved_count / max(1, self.num_incidents)
        awt_pct = self.awaiting_approval_count / max(1, self.num_incidents)
        esc_pct = self.escalated_count / max(1, self.num_incidents)

        return f"""=== NirnayaX End-to-End Evaluation Report ===
Incidents Evaluated : {self.num_incidents}

--- ML Triage Accuracy ---
Category Accuracy   : {self.ml_category_acc:.1%}
Subcategory Acc     : {self.ml_subcategory_acc:.1%}
Priority Accuracy   : {self.ml_priority_acc:.1%}

--- Hybrid RAG Retrieval ---
Runbook Recall@1    : {self.retrieval_recall_at_1:.1%}
Runbook Recall@3    : {self.retrieval_recall_at_3:.1%}
Runbook MRR         : {self.retrieval_mrr:.3f}

--- Workflow & Gating Distribution ---
Resolved Incidents  : {self.resolved_count} ({res_pct:.1%})
Awaiting Approval   : {self.awaiting_approval_count} ({awt_pct:.1%})
Escalated Incidents : {self.escalated_count} ({esc_pct:.1%})
Auto-Approved Count : {self.auto_approved_count}

--- Latency Benchmarks ---
ML Triage Latency   : mean={self.triage_mean_ms:.2f}ms, p95={self.triage_p95_ms:.2f}ms
RAG Retrieval       : mean={self.retrieval_mean_ms:.2f}ms, p95={self.retrieval_p95_ms:.2f}ms
E2E Workflow        : mean={self.workflow_mean_ms:.2f}ms, p95={self.workflow_p95_ms:.2f}ms
"""


class E2EEvaluator:
    """Evaluates the complete NirnayaX system pipeline end-to-end."""

    def __init__(
        self,
        model: TriageModel,
        dataset: IncidentDataset,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.runbooks = build_runbooks()
        self.rb_retriever = build_runbook_retriever(self.runbooks)
        self.inc_retriever = build_incident_retriever(self.dataset.incidents)

    def evaluate(self, sample_size: int = 100) -> E2EEvaluationReport:
        """Run evaluation over a sample of incidents and collect metrics."""
        incidents = self.dataset.incidents[:sample_size]
        n = len(incidents)

        cat_hits = 0
        subcat_hits = 0
        prio_hits = 0

        rec_at_1 = 0
        rec_at_3 = 0
        rr_sum = 0.0

        resolved_cnt = 0
        awaiting_cnt = 0
        escalated_cnt = 0
        auto_appr_cnt = 0

        triage_latencies: list[float] = []
        retrieval_latencies: list[float] = []
        workflow_latencies: list[float] = []

        env = SimulatedEnvironment()
        jira = MockJiraAdapter()
        engine = GuardrailEngine()

        workflow = DiagnosisWorkflow(
            model=self.model,
            runbook_retriever=self.rb_retriever,
            incident_retriever=self.inc_retriever,
            env=env,
            jira_adapter=jira,
            guardrail_engine=engine,
            auto_approve_low_risk=True,
        )

        for inc in incidents:
            draft = TicketDraft(
                title=inc.title,
                description=inc.description,
                affected_service=inc.affected_service,
                region=inc.region,
                channel=inc.channel,
                tags=inc.tags,
            )

            # 1. Measure ML Triage
            t0 = time.perf_counter()
            pred = self.model.predict(draft)
            t1 = time.perf_counter()
            triage_latencies.append((t1 - t0) * 1000.0)

            if pred.category.label == inc.category.value:
                cat_hits += 1
            if pred.subcategory.label == inc.subcategory.value:
                subcat_hits += 1
            if pred.priority.label == inc.priority.value:
                prio_hits += 1

            # 2. Measure RAG Retrieval
            t2 = time.perf_counter()
            query = f"{draft.title}\n{draft.description}"
            rb_matches = self.rb_retriever.retrieve(query, k=3)
            t3 = time.perf_counter()
            retrieval_latencies.append((t3 - t2) * 1000.0)

            # Check if relevant runbook was retrieved
            found_rank = 0
            for rank, res in enumerate(rb_matches, start=1):
                if res.chunk.metadata.get("subcategory") == inc.subcategory.value:
                    found_rank = rank
                    break

            if found_rank == 1:
                rec_at_1 += 1
            if found_rank > 0 and found_rank <= 3:
                rec_at_3 += 1
            if found_rank > 0:
                rr_sum += 1.0 / found_rank

            # 3. Measure Workflow Execution
            t4 = time.perf_counter()
            state = workflow.run(draft, auto_approve_pending=False)
            t5 = time.perf_counter()
            workflow_latencies.append((t5 - t4) * 1000.0)

            if state.status == WorkflowStatus.RESOLVED:
                resolved_cnt += 1
            elif state.status == WorkflowStatus.AWAITING_APPROVAL:
                awaiting_cnt += 1
            elif state.status == WorkflowStatus.ESCALATED:
                escalated_cnt += 1

            if state.approval_request and state.approval_request.status.value == "AUTO_APPROVED":
                auto_appr_cnt += 1

        return E2EEvaluationReport(
            num_incidents=n,
            ml_category_acc=cat_hits / max(1, n),
            ml_subcategory_acc=subcat_hits / max(1, n),
            ml_priority_acc=prio_hits / max(1, n),
            retrieval_recall_at_1=rec_at_1 / max(1, n),
            retrieval_recall_at_3=rec_at_3 / max(1, n),
            retrieval_mrr=rr_sum / max(1, n),
            resolved_count=resolved_cnt,
            awaiting_approval_count=awaiting_cnt,
            escalated_count=escalated_cnt,
            auto_approved_count=auto_appr_cnt,
            triage_mean_ms=float(np.mean(triage_latencies)),
            triage_p95_ms=float(np.percentile(triage_latencies, 95)),
            retrieval_mean_ms=float(np.mean(retrieval_latencies)),
            retrieval_p95_ms=float(np.percentile(retrieval_latencies, 95)),
            workflow_mean_ms=float(np.mean(workflow_latencies)),
            workflow_p95_ms=float(np.percentile(workflow_latencies, 95)),
        )

