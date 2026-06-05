import unittest
from datetime import datetime, timedelta, timezone
from src.nirnayax.domain.sla import calculate_sla_deadline, is_sla_breached

class TestDomainSLA(unittest.TestCase):
    def test_p0_sla_deadline(self):
        start = datetime(2026, 6, 5, 10, 0, 0, tzinfo=timezone.utc)
        deadline = calculate_sla_deadline("P0", start)
        self.assertEqual(deadline, start + timedelta(minutes=15))

    def test_sla_breach_detection(self):
        start = datetime(2026, 6, 5, 10, 0, 0, tzinfo=timezone.utc)
        deadline = calculate_sla_deadline("P1", start)
        
        # 20 mins later: not breached
        self.assertFalse(is_sla_breached(deadline, start + timedelta(minutes=20)))
        # 35 mins later: breached
        self.assertTrue(is_sla_breached(deadline, start + timedelta(minutes=35)))

if __name__ == '__main__':
    unittest.main()
