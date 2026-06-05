import unittest

TIER_MULTIPLIERS = {"ENTERPRISE_PLATINUM": 0.5, "BUSINESS_GOLD": 0.75, "STANDARD": 1.0}

def get_adjusted_sla(base_minutes, tier):
    mult = TIER_MULTIPLIERS.get(tier, 1.0)
    return int(base_minutes * mult)

class TestTierSLA(unittest.TestCase):
    def test_multiplier(self):
        self.assertEqual(get_adjusted_sla(60, "ENTERPRISE_PLATINUM"), 30)
        self.assertEqual(get_adjusted_sla(60, "STANDARD"), 60)
