import unittest
from src.nirnayax.jira.rate_limiter import execute_with_backoff

class TestJiraRateLimiter(unittest.TestCase):
    def test_immediate_success(self):
        res = execute_with_backoff(lambda: "created_issue_123")
        self.assertEqual(res, "created_issue_123")

    def test_retry_on_rate_limit(self):
        attempts = 0
        def flaky_api():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise RuntimeError("429 Too Many Requests")
            return "recovered"

        res = execute_with_backoff(flaky_api, max_retries=3, initial_delay=0.01)
        self.assertEqual(res, "recovered")
        self.assertEqual(attempts, 2)

if __name__ == '__main__':
    unittest.main()
