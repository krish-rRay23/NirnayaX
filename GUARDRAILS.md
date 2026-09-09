# NirnayaX Security Guardrails & Safety Architecture

This document describes the security posture, prompt injection protections, PII sanitization, tool allowlisting, Jira safety rules, and audit trail mechanisms implemented in **NirnayaX**.

---

## 1. Security Principles

1. **Policy-Driven Centralization**: All safety and policy rules are managed by `GuardrailEngine` and `GuardrailPolicyConfig` rather than scattered hard-coded checks.
2. **Fail-Closed Design**: Any security exception, prompt injection payload, unallowlisted tool call, or policy violation immediately blocks execution and escalates the incident to human operators.
3. **Zero Secret Exposure**: Credentials, API tokens (`sk-...`), and passwords are masked in logs, Jira comments, and string representations using Pydantic `SecretStr` and regex masking.

---

## 2. Security Capabilities Matrix

| Guardrail Layer | Component | Functionality |
| :--- | :--- | :--- |
| **Input Validation** | `InputSanitizer` | Validates title/description length, normalizes unicode, strips malicious control characters. |
| **Prompt Injection** | `PromptInjectionDetector` | Scans for instruction overrides ("ignore previous instructions", "system prompt override"), script tags (`<script>`), and command injection (`rm -rf`, `DROP TABLE`). |
| **PII & Secrets** | `PIISanitizer` | Redacts IPv4/IPv6 addresses, API keys (`sk-...`, `Bearer ...`), passwords, emails, SSNs, and credit cards. |
| **Tool Allowlist** | `GuardrailEngine` | Restricts diagnostic tools to `check_service_metrics` / `fetch_service_logs` and remediation actions to explicit allowlists. |
| **Human Approval Gate**| `ApprovalGuardrail` | Strictly requires explicit human approval for `CRITICAL/HIGH` risk actions or `P1/SEV1` incidents. **Critical incidents are never auto-closed**. |
| **Jira Safety** | `GuardrailEngine` | Blocks destructive Jira operations (`delete_issue`, `delete_comment`) and unsafe transitions. |
| **Audit Logging** | `AuditLogger` | Generates immutable, structured audit event records with `trace_id`, `actor`, `decision`, `confidence`, `evidence`, and `status`. |

---

## 3. Threat Model & Mitigation Summary

| Threat | Mitigation in NirnayaX |
| :--- | :--- |
| **Adversarial Prompt Injection** | `PromptInjectionDetector` intercepts injection patterns; workflow fails closed to `ESCALATED`. |
| **Credential Leakage in Logs/Jira** | `PIISanitizer` automatically masks API keys (`[REDACTED_API_KEY]`) and passwords (`[REDACTED_SECRET]`). |
| **Unintended High-Risk Remediation** | `create_approval_request` classifies risk and pauses workflow at `AWAITING_APPROVAL`. |
| **Unauthorized Incident Auto-Closure** | `validate_jira_action` prohibits closing P1/Critical tickets without explicit human approval. |
| **Unsafe Diagnostic / Command Execution** | `validate_tool_call` enforces tool & action allowlisting. |

---

## 4. Policy Configuration via Environment Variables

Guardrails can be configured using environment variables:

```bash
# Force fail-closed behavior on guardrail violations
export NIRNAYAX_GUARDRAILS_FAIL_CLOSED=true

# Minimum confidence score required for automated remediation
export NIRNAYAX_MIN_CONFIDENCE_THRESHOLD=0.50

# Strictly prohibit auto-closing Critical/P1 incidents
export NIRNAYAX_PROHIBIT_AUTO_CLOSE_CRITICAL=true
```
