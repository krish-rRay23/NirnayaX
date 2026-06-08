import re
from typing import Dict, Any

IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
CARD_PATTERN = re.compile(r'\b(?:\d[ -]*?){13,16}\b')

def redact_pii_text(text: str) -> str:
    """Redacts IP addresses, email addresses, and credit card patterns from incident text."""
    if not text:
        return ""
    text = EMAIL_PATTERN.sub('[EMAIL_REDACTED]', text)
    text = IP_PATTERN.sub('[IP_REDACTED]', text)
    text = CARD_PATTERN.sub('[CARD_REDACTED]', text)
    return text

def sanitize_ticket_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively redacts PII strings within an incident ticket dictionary."""
    sanitized = {}
    for k, v in payload.items():
        if isinstance(v, str):
            sanitized[k] = redact_pii_text(v)
        elif isinstance(v, dict):
            sanitized[k] = sanitize_ticket_payload(v)
        elif isinstance(v, list):
            sanitized[k] = [redact_pii_text(x) if isinstance(x, str) else x for x in v]
        else:
            sanitized[k] = v
    return sanitized
