"""Thread-safe structured audit logger for security, decisions, and guardrail enforcement."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .types import AuditEvent, AuditEventStatus


class AuditLogger:
    """Structured audit logger storing immutable audit events."""

    def __init__(self, sink_file: Path | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._lock = Lock()
        self.sink_file = sink_file
        if sink_file:
            sink_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        *,
        trace_id: str,
        incident_id: str,
        actor: str,
        decision: str,
        confidence: float,
        evidence: list[str] | None = None,
        tool_action: str,
        guardrail_results: list[dict[str, Any]] | None = None,
        status: AuditEventStatus = AuditEventStatus.ALLOWED,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Create and store a structured audit event."""
        event = AuditEvent(
            trace_id=trace_id,
            incident_id=incident_id,
            actor=actor,
            decision=decision,
            confidence=confidence,
            evidence=evidence or [],
            tool_action=tool_action,
            timestamp=datetime.now(UTC),
            guardrail_results=guardrail_results or [],
            status=status,
            metadata=metadata or {},
        )

        with self._lock:
            self._events.append(event)
            if self.sink_file:
                with self.sink_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event.model_dump(mode="json")) + "\n")

        return event

    def get_events(self, incident_id: str | None = None) -> list[AuditEvent]:
        """Retrieve audit events, optionally filtered by incident_id."""
        with self._lock:
            if incident_id is None:
                return list(self._events)
            return [e for e in self._events if e.incident_id == incident_id]

    def clear(self) -> None:
        """Clear in-memory audit logs."""
        with self._lock:
            self._events.clear()
