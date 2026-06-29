import unittest
from src.nirnayax.metrics import TriageMetricsCollector

class TestMetrics(unittest.TestCase):
    def test_metrics_collection(self):
        collector = TriageMetricsCollector()
        collector.record_triage("GATEWAY_TIMEOUT", "route")
        collector.record_triage("SECURITY_ANOMALY", "escalate")
        collector.record_triage("GATEWAY_TIMEOUT", "route")

        snap = collector.snapshot()
        self.assertEqual(snap["total_incidents"], 3)
        self.assertEqual(snap["auto_routed"], 2)
        self.assertEqual(snap["escalated"], 1)
        self.assertAlmostEqual(snap["automation_rate"], 0.667)
        self.assertEqual(snap["categories"]["GATEWAY_TIMEOUT"], 2)

if __name__ == '__main__':
    unittest.main()
