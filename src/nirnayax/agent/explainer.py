from typing import Dict, Any, List

def format_triage_explanation(
    category: str,
    confidence: float,
    evidence_signals: List[str],
    gating_action: str
) -> Dict[str, Any]:
    """Constructs a transparent audit explanation for L1 triage gating."""
    return {
        "summary": f"Incident classified as '{category}' with {round(confidence * 100, 1)}% confidence.",
        "gating_decision": gating_action,
        "is_autonomous": gating_action.lower() in ("auto_resolve", "auto_execute", "route"),
        "key_signals": evidence_signals,
        "review_required": confidence < 0.75 or gating_action.lower() == "escalate"
    }
