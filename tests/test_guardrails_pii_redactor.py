import unittest
from src.nirnayax.guardrails.pii_redactor import redact_pii_text, sanitize_ticket_payload

class TestGuardrailsPIIRedactor(unittest.TestCase):
    def test_email_and_ip_redaction(self):
        raw = "User admin@telecom.net reported outage at 192.168.1.100 gateway"
        redacted = redact_pii_text(raw)
        self.assertNotIn("admin@telecom.net", redacted)
        self.assertNotIn("192.168.1.100", redacted)
        self.assertIn("[EMAIL_REDACTED]", redacted)
        self.assertIn("[IP_REDACTED]", redacted)

    def test_sanitize_payload(self):
        payload = {
            "title": "Alert for server 10.0.0.1",
            "details": {
                "owner": "support@infra.org",
                "code": 500
            }
        }
        clean = sanitize_ticket_payload(payload)
        self.assertEqual(clean["title"], "Alert for server [IP_REDACTED]")
        self.assertEqual(clean["details"]["owner"], "[EMAIL_REDACTED]")
        self.assertEqual(clean["details"]["code"], 500)

if __name__ == '__main__':
    unittest.main()
