import unittest
from src.nirnayax.agent.explainer import format_triage_explanation

class TestAgentExplainer(unittest.TestCase):
    def test_high_confidence_auto_route(self):
        res = format_triage_explanation(
            category="GATEWAY_TIMEOUT",
            confidence=0.92,
            evidence_signals=["504 Gateway Timeout in upstream log"],
            gating_action="route"
        )
        self.assertTrue(res["is_autonomous"])
        self.assertFalse(res["review_required"])
        self.assertIn("92.0%", res["summary"])

    def test_low_confidence_requires_review(self):
        res = format_triage_explanation(
            category="UNKNOWN",
            confidence=0.45,
            evidence_signals=["Ambiguous traceback"],
            gating_action="escalate"
        )
        self.assertFalse(res["is_autonomous"])
        self.assertTrue(res["review_required"])

if __name__ == '__main__':
    unittest.main()
