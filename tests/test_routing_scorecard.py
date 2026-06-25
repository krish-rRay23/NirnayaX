import unittest

def calculate_scorecard(correct_routes, total_routes):
    return (correct_routes / total_routes) * 100.0 if total_routes > 0 else 0.0

class TestScorecard(unittest.TestCase):
    def test_score(self):
        self.assertEqual(calculate_scorecard(95, 100), 95.0)
