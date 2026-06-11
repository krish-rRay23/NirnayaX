import unittest

def ensemble_vote(votes):
    tally = {}
    for choice, weight in votes:
        tally[choice] = tally.get(choice, 0.0) + weight
    return max(tally.items(), key=lambda x: x[1])[0]

class TestVoter(unittest.TestCase):
    def test_voting(self):
        winner = ensemble_vote([("INFRA", 0.6), ("APP", 0.3), ("INFRA", 0.5)])
        self.assertEqual(winner, "INFRA")
