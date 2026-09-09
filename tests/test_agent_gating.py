"""Tests for confidence and evidence gating."""

from __future__ import annotations

from nirnayax.agent.gating import evaluate_confidence_gate
from nirnayax.agent.types import DecisionType, Observation
from nirnayax.ml.types import ClassPrediction, TriagePrediction
from nirnayax.retrieval import RetrievalResult
from nirnayax.retrieval.types import Chunk


def _mock_prediction(
    subcat: str = "BGP_ROUTING", conf: float = 0.90, consistent: bool = True
) -> TriagePrediction:
    net_subs = ("BGP_ROUTING", "LINK_DOWN", "LATENCY_PACKET_LOSS", "DNS_RESOLUTION")
    cat = "NETWORK" if subcat in net_subs else "APPLICATION_DB"
    return TriagePrediction(
        category=ClassPrediction(label=cat, confidence=conf, distribution=((cat, conf),)),
        subcategory=ClassPrediction(label=subcat, confidence=conf, distribution=((subcat, conf),)),
        priority=ClassPrediction(label="P1", confidence=0.85, distribution=(("P1", 0.85),)),
        taxonomy_consistent=consistent,
        model_version="1.0.0",
    )


def _mock_result(subcat: str, score: float = 0.8) -> RetrievalResult:
    chunk = Chunk(
        chunk_id="RB-1#0",
        doc_id="RB-1",
        index=0,
        text="Sample text",
        start=0,
        end=10,
        source_type="runbook",
        source_id="RB-1",
        title="Sample Runbook",
        metadata={"subcategory": subcat},
    )
    return RetrievalResult(chunk=chunk, score=score, rank=1)


def test_confidence_gate_remediate_when_all_evidence_aligns() -> None:
    pred = _mock_prediction("BGP_ROUTING", conf=0.90)
    runbooks = [_mock_result("BGP_ROUTING", score=0.8)]
    incidents = [_mock_result("BGP_ROUTING", score=0.7)]
    observations = [
        Observation(
            tool_name="m",
            target_service="s",
            metric_name="flaps",
            value=20.0,
            passed=False,
            details="Anomaly observed",
        )
    ]

    outcome = evaluate_confidence_gate(
        prediction=pred,
        retrieved_runbooks=runbooks,
        similar_incidents=incidents,
        observations=observations,
        min_threshold=0.65,
    )

    assert outcome.decision == DecisionType.REMEDIATE
    assert outcome.confidence_score >= 0.65
    assert "High confidence" in outcome.reasoning


def test_confidence_gate_escalates_on_low_ml_confidence() -> None:
    pred = _mock_prediction("BGP_ROUTING", conf=0.20)
    runbooks = [_mock_result("BGP_ROUTING", score=0.8)]
    incidents = [_mock_result("BGP_ROUTING", score=0.7)]
    observations = [
        Observation(
            tool_name="m",
            target_service="s",
            metric_name="flaps",
            value=20.0,
            passed=False,
            details="Anomaly observed",
        )
    ]

    outcome = evaluate_confidence_gate(
        prediction=pred,
        retrieved_runbooks=runbooks,
        similar_incidents=incidents,
        observations=observations,
        min_threshold=0.65,
    )

    assert outcome.decision == DecisionType.ESCALATE
    assert "ML triage confidence too low" in outcome.reasoning


def test_confidence_gate_escalates_on_mismatched_runbook() -> None:
    pred = _mock_prediction("BGP_ROUTING", conf=0.90)
    runbooks = [_mock_result("DISK_SPACE", score=0.8)]
    incidents = [_mock_result("BGP_ROUTING", score=0.7)]
    observations = [
        Observation(
            tool_name="m",
            target_service="s",
            metric_name="flaps",
            value=20.0,
            passed=False,
            details="Anomaly observed",
        )
    ]

    outcome = evaluate_confidence_gate(
        prediction=pred,
        retrieved_runbooks=runbooks,
        similar_incidents=incidents,
        observations=observations,
        min_threshold=0.65,
    )

    assert outcome.decision == DecisionType.ESCALATE
    assert "does not match" in outcome.reasoning


def test_confidence_gate_escalates_when_no_diagnostic_anomaly() -> None:
    pred = _mock_prediction("BGP_ROUTING", conf=0.90)
    runbooks = [_mock_result("BGP_ROUTING", score=0.8)]
    incidents = [_mock_result("BGP_ROUTING", score=0.7)]
    observations = [
        Observation(
            tool_name="m",
            target_service="s",
            metric_name="flaps",
            value=0.0,
            passed=True,
            details="Normal",
        )
    ]

    outcome = evaluate_confidence_gate(
        prediction=pred,
        retrieved_runbooks=runbooks,
        similar_incidents=incidents,
        observations=observations,
        min_threshold=0.65,
    )

    assert outcome.decision == DecisionType.ESCALATE
    assert "no clear anomaly" in outcome.reasoning
