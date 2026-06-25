import unittest

def calculate_average_resolution(durations_list):
    return sum(durations_list) / len(durations_list) if durations_list else 0.0

class TestDuration(unittest.TestCase):
    def test_avg(self):
        self.assertEqual(calculate_average_resolution([10, 20, 30]), 20.0)
