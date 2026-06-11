import unittest
import json

def export_rationale(incident_id, chosen_category, confidence, factors):
    return json.dumps({
        "incident_id": incident_id,
        "category": chosen_category,
        "confidence": confidence,
        "factors": factors
    })

class TestRationale(unittest.TestCase):
    def test_json(self):
        data = export_rationale("INC-101", "DATABASE", 0.94, ["slow_query", "cpu_99"])
        self.assertIn("DATABASE", data)
