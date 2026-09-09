"""Tests for PIISanitizer."""

from __future__ import annotations

from nirnayax.guardrails import PIISanitizer
from nirnayax.ml import TicketDraft


def test_pii_sanitizer_redacts_api_keys() -> None:
    text = "Error connecting with API key sk-1234567890123456789012345678 and Bearer abc123def456=="
    sanitized, count = PIISanitizer.sanitize_text(text)
    assert "sk-12345" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
    assert count >= 1


def test_pii_sanitizer_redacts_passwords_and_emails() -> None:
    text = "User admin@company.internal failed login with password='SuperSecretPassword123!'"
    sanitized, count = PIISanitizer.sanitize_text(text)
    assert "SuperSecretPassword123" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized
    assert "admin@company.internal" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized
    assert count >= 2


def test_pii_sanitizer_redacts_ip_addresses() -> None:
    text = "Connection failure from 192.168.1.100 to target 10.0.0.1"
    sanitized, count = PIISanitizer.sanitize_text(text)
    assert "192.168.1.100" not in sanitized
    assert "[REDACTED_IP]" in sanitized
    assert count == 2


def test_pii_sanitizer_ticket_draft() -> None:
    draft = TicketDraft(
        title="Flapping BGP on 172.16.0.1",
        description="Contact engineer@corp.com with secret=TopSecretKey99",
        affected_service="core-router",
    )
    sanitized_draft, count = PIISanitizer.sanitize_draft(draft)
    assert "172.16.0.1" not in sanitized_draft.title
    assert "engineer@corp.com" not in sanitized_draft.description
    assert "TopSecretKey99" not in sanitized_draft.description
    assert count >= 3
