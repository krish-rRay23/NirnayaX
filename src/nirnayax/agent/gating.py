"""Confidence and evidence gating for agentic remediation vs escalation decisions."""

from __future__ import annotations

from collections.abc import Sequence

from ..ml import TriagePrediction
from ..retrieval import RetrievalResult
from .types import DecisionOutcome, DecisionType, Observation

MIN_CONFIDENCE_THRESHOLD = 0.65


def evaluate_confidence_gate(
    *,
    prediction: TriagePrediction | None,
    retrieved_runbooks: Sequence[RetrievalResult],
    similar_incidents: Sequence[RetrievalResult],
    observations: Sequence[Observation],
    min_threshold: float = MIN_CONFIDENCE_THRESHOLD,
) -> DecisionOutcome:
    """Evaluate whether evidence is sufficient to trigger automated remediation or escalate."""

    reasons: list[str] = []

    # 1. ML triage confidence
    if prediction is None:
        return DecisionOutcome(
            decision=DecisionType.ESCALATE,
            confidence_score=0.0,
            reasoning="No ML triage prediction available.",
            suggested_action="Escalate to L2 team for manual triage.",
        )

    ml_subcat = prediction.subcategory.label
    ml_conf = prediction.subcategory.confidence
    if not prediction.taxonomy_consistent:
        ml_conf *= 0.5
        reasons.append("Taxonomy inconsistency between category and subcategory.")
    if ml_conf < 0.40:
        reasons.append(f"ML triage confidence too low ({ml_conf:.1%}).")

    # 2. Runbook match evidence
    matching_runbook = None
    runbook_score = 0.0
    if retrieved_runbooks:
        top_rb = retrieved_runbooks[0]
        if top_rb.chunk.metadata.get("subcategory") == ml_subcat:
            matching_runbook = top_rb
            # Map RRF / rerank score to normalized 0..1 scale
            runbook_score = min(1.0, max(0.5, top_rb.score * 3.0))
        else:
            reasons.append(
                f"Top runbook ({top_rb.chunk.metadata.get('subcategory')}) "
                f"does not match predicted subcategory ({ml_subcat})."
            )
    else:
        reasons.append("No relevant runbook retrieved.")

    # 3. Historical incident correlation evidence
    incident_match_ratio = 0.0
    if similar_incidents:
        matches = sum(
            1 for inc in similar_incidents[:3] if inc.chunk.metadata.get("subcategory") == ml_subcat
        )
        incident_match_ratio = matches / min(3, len(similar_incidents))
    else:
        reasons.append("No historical incidents correlated.")

    # 4. Diagnostic observation confirmation
    diagnostic_confirmed = any(not obs.passed for obs in observations)
    if not diagnostic_confirmed:
        reasons.append("Diagnostic metrics showed no clear anomaly.")

    # Composite confidence calculation
    composite_score = (
        0.40 * ml_conf
        + 0.30 * runbook_score
        + 0.20 * incident_match_ratio
        + 0.10 * (1.0 if diagnostic_confirmed else 0.0)
    )

    # Gate decision rule
    can_remediate = (
        composite_score >= min_threshold
        and ml_conf >= 0.40
        and matching_runbook is not None
        and diagnostic_confirmed
    )

    if can_remediate:
        rb_title = matching_runbook.chunk.title if matching_runbook else ml_subcat
        rb_id = matching_runbook.chunk.source_id if matching_runbook else ""
        reasoning = (
            f"High confidence ({composite_score:.1%}) triage for {ml_subcat}. "
            f"Diagnostic metrics confirmed anomaly; runbook '{rb_title}' gives resolution steps."
        )
        return DecisionOutcome(
            decision=DecisionType.REMEDIATE,
            confidence_score=composite_score,
            reasoning=reasoning,
            suggested_action=f"Execute automated remediation for {ml_subcat} per runbook {rb_id}",
        )

    # Escalation fall-through
    escalation_detail = "; ".join(reasons) if reasons else "Insufficient overall confidence."
    reasoning = (
        f"Confidence ({composite_score:.1%}) below threshold ({min_threshold:.1%}) "
        f"or evidence gate failed: {escalation_detail}"
    )
    return DecisionOutcome(
        decision=DecisionType.ESCALATE,
        confidence_score=composite_score,
        reasoning=reasoning,
        suggested_action=(
            f"Escalate {ml_subcat} ticket to on-call engineering team for manual investigation."
        ),
    )


__all__ = ["MIN_CONFIDENCE_THRESHOLD", "evaluate_confidence_gate"]
