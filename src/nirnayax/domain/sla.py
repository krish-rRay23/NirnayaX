from typing import Dict, Any
from datetime import datetime, timedelta, timezone

SLA_MINUTES = {
    "P0": 15,
    "P1": 30,
    "P2": 120,
    "P3": 1440,
}

def calculate_sla_deadline(severity: str, reported_at: datetime = None) -> datetime:
    """Calculates resolution deadline timestamp based on severity tier."""
    base_time = reported_at or datetime.now(timezone.utc)
    minutes = SLA_MINUTES.get(severity.upper(), 1440)
    return base_time + timedelta(minutes=minutes)

def is_sla_breached(deadline: datetime, current_time: datetime = None) -> bool:
    """Checks whether the current time exceeds the resolution deadline."""
    now = current_time or datetime.now(timezone.utc)
    return now > deadline
