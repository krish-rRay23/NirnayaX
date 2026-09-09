"""Tests for AuditLogger."""

from __future__ import annotations

from nirnayax.guardrails import AuditEventStatus, AuditLogger


def test_audit_logger_records_structured_events() -> None:
    logger = AuditLogger()
    event = logger.log_event(
        trace_id="TRC-TEST-001",
        incident_id="INC-101",
        actor="GuardrailEngine",
        decision="REMEDIATE",
        confidence=0.92,
        evidence=["RB-NETWORK-004"],
        tool_action="reset_bgp_peering_session",
        status=AuditEventStatus.ALLOWED,
    )

    assert event.trace_id == "TRC-TEST-001"
    assert event.incident_id == "INC-101"
    assert event.actor == "GuardrailEngine"
    assert event.confidence == 0.92
    assert event.status == AuditEventStatus.ALLOWED

    events = logger.get_events("INC-101")
    assert len(events) == 1
    assert events[0].incident_id == "INC-101"
    rendered = event.render()
    assert "TRC-TEST-001" in rendered
    assert "INC-101" in rendered
