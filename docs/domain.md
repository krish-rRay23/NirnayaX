# Domain model

The domain layer is the **contract** every other layer depends on. It has two
modules:

- [`nirnayax.domain.taxonomy`](../src/nirnayax/domain/taxonomy.py) — the
  controlled vocabularies (what labels exist).
- [`nirnayax.domain.models`](../src/nirnayax/domain/models.py) — the typed
  records (what a valid object looks like).

Two design rules run through both:

1. **The taxonomy is the single source of truth.** Categories and subcategories
   live in one data structure; generation, validation, and EDA all derive from
   it. An import-time check fails fast if the enum and the mapping ever drift.
2. **Models are values, not bags.** Every model is `frozen=True` (immutable,
   hashable) and `extra="forbid"` (unknown fields are rejected). Invariants are
   enforced at construction, so an object that exists is an object that is valid.

---

## Taxonomy

### Categories → subcategories

| Category | Subcategories |
| --- | --- |
| `NETWORK` | `LATENCY_PACKET_LOSS`, `LINK_DOWN`, `DNS_RESOLUTION`, `BGP_ROUTING` |
| `APPLICATION_DB` | `CONNECTION_POOL_EXHAUSTION`, `REPLICATION_LAG`, `DEADLOCK`, `SLOW_QUERY`, `DISK_SPACE` |
| `BILLING_OSS` | `RATING_ENGINE_ERROR`, `INVOICE_GENERATION_FAILURE`, `MEDIATION_FEED_GAP`, `PROVISIONING_SYNC_FAILURE` |
| `HARDWARE_ACCESS` | `ACCOUNT_LOCKOUT`, `VPN_ACCESS_FAILURE`, `SERVER_HARDWARE_FAULT`, `PERIPHERAL_FAILURE` |

4 categories, 17 subcategories. Each subcategory maps to **exactly one**
category.

### Operational enums

| Enum | Values | Meaning |
| --- | --- | --- |
| `Severity` | `SEV1` … `SEV4` | Technical impact (SEV1 = most severe). |
| `Priority` | `P1` … `P4` | Work-queue ordering (P1 = handle first). |
| `IncidentStatus` | `NEW`, `TRIAGED`, `IN_PROGRESS`, `RESOLVED`, `CLOSED` | Ticket lifecycle. |
| `Channel` | `MONITORING`, `EMAIL`, `PHONE`, `CHAT`, `SELF_SERVICE` | How the incident entered the queue. |

All are `StrEnum`, so their JSON form is the plain string (`"SEV1"`), and they
compare equal to that string — convenient for serialization and dataframes.

### Helper API

```python
from nirnayax.domain import (
    TAXONOMY,
    category_of,
    subcategories_of,
    all_subcategories,
)

category_of(Subcategory.DEADLOCK)  # -> Category.APPLICATION_DB
subcategories_of(Category.NETWORK)  # -> (LATENCY_PACKET_LOSS, LINK_DOWN, ...)
all_subcategories()  # -> every subcategory, in taxonomy order
```

### Extending the taxonomy

To add a failure mode: add the value to the `Subcategory` enum **and** to the
`TAXONOMY` mapping under its category. That's the only change required — the
import-time `_validate_taxonomy()` guard guarantees the two never fall out of
sync (it raises if a subcategory is unmapped or duplicated). Adding a template in
[`catalog.py`](../src/nirnayax/data/catalog.py) and a runbook spec in
[`runbooks.py`](../src/nirnayax/data/runbooks.py) then lights it up in generation
and the knowledge base.

---

## Models

### `Signal`
A single observability datapoint attached to an incident.

| Field | Type | Constraints |
| --- | --- | --- |
| `name` | `str` | 1–64 chars |
| `value` | `float` | — |
| `unit` | `str` | ≤ 16 chars (e.g. `"ms"`, `"%"`, `"count"`) |

### `Incident`
A single L1 incident record — the central object of the system.

| Field | Type | Notes |
| --- | --- | --- |
| `incident_id` | `str` | matches `^INC-\d{6}$` |
| `title` | `str` | 3–160 chars |
| `description` | `str` | 10–2000 chars |
| `category` | `Category` | ground-truth routing label |
| `subcategory` | `Subcategory` | must belong to `category` (validated) |
| `severity` | `Severity` | |
| `priority` | `Priority` | |
| `status` | `IncidentStatus` | defaults to `NEW` |
| `channel` | `Channel` | |
| `affected_service` | `str` | 1–80 chars |
| `region` | `str` | 1–40 chars |
| `reported_at` | `datetime` | timezone-aware |
| `signals` | `tuple[Signal, ...]` | defaults to `()` |
| `tags` | `tuple[str, ...]` | defaults to `()` |
| `customer_impacting` | `bool` | defaults to `False` |

A model validator enforces `category == category_of(subcategory)` — a
mislabelled incident cannot be constructed.

### `Runbook` and `RunbookStep`
A synthetic troubleshooting guide (the future retrieval knowledge base). Each
runbook targets one subcategory and carries operational metadata.

`RunbookStep`: `order` (≥ 1), `action` (1–400 chars), optional `expected_signal`.

`Runbook` key fields: `runbook_id` (`^RB-[A-Z_]+-\d{3}$`), `title`, `category`
/`subcategory` (consistency-checked), `summary`, `symptoms` (≥ 1), `steps`
(≥ 1), `escalation_team`, `severity_hint`, `estimated_resolution_minutes`
(1–10080), `tags`, `version` (semver), `last_reviewed`, `related_runbook_ids`.
A validator enforces that `steps` are numbered contiguously `1..n`.

### Dataset containers

- `DatasetSplit` — `train` | `eval`.
- `DatasetMetadata` — provenance: `name`, `split`, `seed`, `size`,
  `generated_at`, `generator_version`, `schema_version`. Enables reproducibility
  auditing.
- `IncidentDataset` — `metadata` + `incidents`; a validator enforces
  `metadata.size == len(incidents)`.

---

## Why immutability + strictness

The data foundation is trusted by everything above it. Making models frozen and
strict means:

- **No silent corruption** — you cannot mutate a validated record or smuggle in
  an unexpected field.
- **Hashable values** — incidents/runbooks can go straight into sets and dict
  keys, and equality is structural (used throughout the determinism tests).
- **Fail at the boundary** — bad data raises where it's constructed, not three
  layers deep.
