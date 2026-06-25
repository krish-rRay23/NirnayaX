# Prometheus Alert Rules

```yaml
- alert: UnassignedIncidentBacklogHigh
  expr: nirnaya_unassigned_incidents_total > 50
  for: 10m
  labels:
    severity: critical
```
