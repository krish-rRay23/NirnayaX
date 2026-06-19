# Blameless Incident Postmortem Template

## 1. Incident Overview
- **Incident ID:** INC-XXXXX
- **Severity:** P0 / P1 / P2
- **Lead Responder:** 
- **Time to Detect (TTD):** 
- **Time to Mitigate (TTM):** 

## 2. Root Cause Summary
Explain the underlying breakdown (e.g. unindexed query burst, DNS TTL expiry, deployment deadlock).

## 3. Timeline
- `10:00 UTC` - Anomaly detected by NirnayaX triage engine.
- `10:02 UTC` - Automated runbook triggered (circuit breaker opened).
- `10:15 UTC` - Human review gate verified failover status.
- `10:30 UTC` - Service traffic restored.

## 4. Corrective Action Items
- [ ] Add rate limiter to upstream webhook endpoint.
- [ ] Adjust policy gating confidence threshold to 0.80.
