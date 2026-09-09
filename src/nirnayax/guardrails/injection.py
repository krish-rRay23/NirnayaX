"""Detector for prompt injection, adversarial payloads, and instruction overrides."""

from __future__ import annotations

import re

from .policy import GuardrailPolicyConfig


class PromptInjectionDetector:
    """Detects prompt injection attempts, system instruction overrides, and command injection."""

    def __init__(self, policy: GuardrailPolicyConfig | None = None) -> None:
        self.policy = policy or GuardrailPolicyConfig()
        # Compile patterns for case-insensitive substring/regex matching
        self._compiled_patterns = [
            re.compile(re.escape(pattern), re.IGNORECASE)
            for pattern in self.policy.injection_patterns
        ]

    def detect(self, text: str) -> tuple[bool, str]:
        """Scans text for prompt injection or command injection attempts.

        Returns (is_injection_detected: bool, reason: str).
        """
        if not text:
            return False, "Empty input text."

        lowered_text = text.lower()

        # Check policy injection patterns
        for pattern_str in self.policy.injection_patterns:
            if pattern_str.lower() in lowered_text:
                msg = (
                    "Prompt injection or dangerous payload detected: "
                    f"matches pattern '{pattern_str}'."
                )
                return True, msg

        # Check suspicious system prompt markers or HTML/Script tags
        if "<script>" in lowered_text or "javascript:" in lowered_text:
            return True, "XSS or script injection payload detected in input."

        if "system:" in lowered_text and "override" in lowered_text:
            return True, "Adversarial system prompt override instruction detected."

        return False, "No prompt injection detected."
