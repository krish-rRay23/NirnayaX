# Agent Autonomous Escalation Thresholds

- Confidence >= 0.85: Direct autonomous triage and Jira assignment.
- Confidence 0.60 - 0.84: Route to Human-in-the-Loop review queue.
- Confidence < 0.60: Fallback to Tier 2 on-call engineer page.
