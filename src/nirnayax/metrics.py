from typing import Dict, Any

class TriageMetricsCollector:
    """In-memory telemetry collector recording triage counters and routing outcomes."""
    def __init__(self):
        self.total_incidents = 0
        self.auto_routed = 0
        self.escalated = 0
        self.category_counts: Dict[str, int] = {}

    def record_triage(self, category: str, action: str):
        self.total_incidents += 1
        self.category_counts[category] = self.category_counts.get(category, 0) + 1
        if action.lower() in ("auto_resolve", "auto_execute", "route"):
            self.auto_routed += 1
        else:
            self.escalated += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_incidents": self.total_incidents,
            "auto_routed": self.auto_routed,
            "escalated": self.escalated,
            "automation_rate": round(self.auto_routed / max(self.total_incidents, 1), 3),
            "categories": self.category_counts
        }
