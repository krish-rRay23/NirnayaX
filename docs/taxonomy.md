# NirnayaX Incident Taxonomy & Root-Cause Matrix

## 1. Domain Classifications
- **NETWORK_OUTAGE**: Backbone routing failures, BGP drops, packet loss spikes (>5%).
- **GATEWAY_TIMEOUT**: Upstream microservice / 3rd-party payment gateway HTTP 504 / connection timeouts.
- **DATABASE_CONTENTION**: Connection pool exhaustion, lock contention, slow query degradation.
- **SECURITY_ANOMALY**: Rate limit bursts, credential stuffing attempts, abnormal token revocation.
- **INFRASTRUCTURE_DEGRADATION**: High CPU/Memory throttling, disk I/O saturation, pod evictions.

## 2. Severity Levels & Escalation
| Severity | Description | Max Response Time | Automated Action Allowed |
|---|---|---|---|
| P0 / Critical | Customer-impacting outage | 15 mins | Bounded restart / fallback |
| P1 / High | Degraded core performance | 30 mins | Cache flush / rate limit |
| P2 / Medium | Isolated feature failure | 2 hours | Ticket routing / Nudge |
| P3 / Low | Minor non-impacting error | 24 hours | Log capture / auto-triage |
