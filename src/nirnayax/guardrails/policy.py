"""Configurable policy rules and parameters for NirnayaX guardrails."""

from __future__ import annotations

import contextlib
import os
from typing import Any

from pydantic import BaseModel


class GuardrailPolicyConfig(BaseModel):
    """Central policy configuration controlling security, safety, and audit rules."""

    enable_input_sanitization: bool = True
    enable_prompt_injection_detection: bool = True
    enable_pii_protection: bool = True
    enable_evidence_enforcement: bool = True
    enable_tool_allowlisting: bool = True
    enable_approval_enforcement: bool = True
    enable_jira_guardrails: bool = True
    fail_closed: bool = True

    # Thresholds
    min_confidence_threshold: float = 0.50
    min_runbook_evidence_count: int = 1
    max_input_length: int = 10000

    # Allowlists
    allowed_tools: tuple[str, ...] = (
        "check_service_metrics",
        "fetch_service_logs",
        "execute_remediation",
        "verify_service_recovery",
    )
    allowed_remediation_actions: tuple[str, ...] = (
        "reset_bgp_peering_session",
        "restart_connection_pool",
        "flush_memcached_cache",
        "rotate_dns_endpoint",
        "rebalance_kafka_partitions",
        "BGP_ROUTING",
        "CONNECTION_POOL_EXHAUSTION",
        "DNS_RESOLUTION",
        "DISK_SPACE",
        "DATABASE_CONCURRENCY",
        "CPU_SPIKE",
        "MEMORY_LEAK",
        "POD_CRASH",
        "SSL_CERT_EXPIRED",
        "CONFIG_SYNTAX_ERROR",
        "NETWORK",
        "DATABASE",
        "APPLICATION",
        "INFRASTRUCTURE",
    )

    # Blocked keywords / injection patterns (case-insensitive substring match)
    injection_patterns: tuple[str, ...] = (
        "ignore previous instructions",
        "ignore all previous instructions",
        "system prompt override",
        "override system prompt",
        "forget all rules",
        "forget your instructions",
        "you are now in admin mode",
        "bypass approval",
        "auto approve critical",
        "auto-approve critical",
        "rm -rf",
        "drop database",
        "drop table",
        "<script>",
        "javascript:",
        "eval(",
        "exec(",
        "delete user",
        "sudo su",
    )

    # Safety controls
    prohibit_auto_close_critical: bool = True
    prohibit_destructive_jira_actions: bool = True

    @classmethod
    def from_env(cls, **overrides: Any) -> GuardrailPolicyConfig:
        """Create policy configuration from environment variables with overrides."""
        env_conf: dict[str, Any] = {}

        if "NIRNAYAX_GUARDRAILS_FAIL_CLOSED" in os.environ:
            env_conf["fail_closed"] = (
                os.environ["NIRNAYAX_GUARDRAILS_FAIL_CLOSED"].lower() in ("true", "1", "yes")
            )

        if "NIRNAYAX_MIN_CONFIDENCE_THRESHOLD" in os.environ:
            with contextlib.suppress(ValueError):
                env_conf["min_confidence_threshold"] = float(
                    os.environ["NIRNAYAX_MIN_CONFIDENCE_THRESHOLD"]
                )

        if "NIRNAYAX_PROHIBIT_AUTO_CLOSE_CRITICAL" in os.environ:
            env_conf["prohibit_auto_close_critical"] = (
                os.environ["NIRNAYAX_PROHIBIT_AUTO_CLOSE_CRITICAL"].lower() in ("true", "1", "yes")
            )

        env_conf.update(overrides)
        return cls(**env_conf)
