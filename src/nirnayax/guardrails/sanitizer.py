"""Sanitizer for PII, sensitive credentials, and raw text normalization."""

from __future__ import annotations

import re

from ..ml import TicketDraft

# Regex patterns for PII and sensitive data
_IPV4_PATTERN = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
_IPV6_PATTERN = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_API_KEY_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._~+/-]+=*)\b",
    re.IGNORECASE,
)
_PASSWORD_PATTERN = re.compile(
    r"(?i)\b(password|passwd|secret|api_key|access_token|auth_token)\s*[:=]\s*['\"]?([^\s'\";,]+)['\"]?"
)


class PIISanitizer:
    """Sanitizes and redacts PII and sensitive credentials from text."""

    @classmethod
    def sanitize_text(cls, text: str) -> tuple[str, int]:
        """Redact PII and sensitive credentials from a text string.

        Returns (sanitized_text, count_of_redactions).
        """
        if not text:
            return text, 0

        redaction_count = 0

        # Redact API Keys / Tokens
        sanitized, count = _API_KEY_PATTERN.subn("[REDACTED_API_KEY]", text)
        redaction_count += count

        # Redact Passwords/Secrets
        def _mask_password(match: re.Match[str]) -> str:
            key = match.group(1)
            return f"{key}=[REDACTED_SECRET]"

        sanitized, count = _PASSWORD_PATTERN.subn(_mask_password, sanitized)
        redaction_count += count

        # Redact SSNs
        sanitized, count = _SSN_PATTERN.subn("[REDACTED_SSN]", sanitized)
        redaction_count += count

        # Redact Emails
        sanitized, count = _EMAIL_PATTERN.subn("[REDACTED_EMAIL]", sanitized)
        redaction_count += count

        # Redact Credit Cards (basic validation length match)
        def _mask_card(match: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", match.group(0))
            if 13 <= len(digits) <= 16:
                return "[REDACTED_CARD]"
            return match.group(0)

        sanitized, count = _CREDIT_CARD_PATTERN.subn(_mask_card, sanitized)
        redaction_count += count

        # Redact IPv4
        sanitized, count = _IPV4_PATTERN.subn("[REDACTED_IP]", sanitized)
        redaction_count += count

        # Redact IPv6
        sanitized, count = _IPV6_PATTERN.subn("[REDACTED_IP]", sanitized)
        redaction_count += count

        return sanitized, redaction_count

    @classmethod
    def sanitize_draft(cls, draft: TicketDraft) -> tuple[TicketDraft, int]:
        """Sanitize all text fields in a TicketDraft object."""
        title_san, c1 = cls.sanitize_text(draft.title)
        desc_san, c2 = cls.sanitize_text(draft.description)
        svc_san, c3 = cls.sanitize_text(draft.affected_service or "")
        reg_san, c4 = cls.sanitize_text(draft.region or "")

        sanitized_tags = tuple(cls.sanitize_text(t)[0] for t in draft.tags)

        sanitized_draft = TicketDraft(
            title=title_san,
            description=desc_san,
            affected_service=svc_san if svc_san else None,
            region=reg_san if reg_san else None,
            channel=draft.channel,
            tags=sanitized_tags,
        )

        total_redactions = c1 + c2 + c3 + c4
        return sanitized_draft, total_redactions
