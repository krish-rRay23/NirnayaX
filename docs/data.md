# Data foundation

The data layer turns the domain contract into concrete, reproducible artifacts:
a synthetic incident corpus, a runbook knowledge base, and the tooling to
validate, explore, and persist them.

- [`generator`](../src/nirnayax/data/generator.py) — deterministic incident synthesis
- [`catalog`](../src/nirnayax/data/catalog.py) — content tables that drive generation
- [`runbooks`](../src/nirnayax/data/runbooks.py) — one troubleshooting runbook per subcategory
- [`validation`](../src/nirnayax/data/validation.py) — collection-level checks
- [`eda`](../src/nirnayax/data/eda.py) — dependency-free summary statistics
- [`serialization`](../src/nirnayax/data/serialization.py) — JSON / JSONL persistence

---

## Determinism

Reproducibility is a hard requirement, so the generator is **fully
deterministic**:

- A single `random.Random(seed)` instance is advanced in a **fixed order** per
  incident. No global RNG, no wall-clock reads.
- `reported_at` is derived from a fixed logical clock
  (`REFERENCE_TIME = 2026-09-01T00:00:00Z`); each incident lands within the
  90 days before it. So serialized timestamps are stable across runs.
- `generated_at` in the metadata is also pinned to `REFERENCE_TIME`.

Consequences (asserted by the test suite): the same `(size, seed)` yields
byte-for-byte identical incidents; different seeds yield different corpora.

### How an incident is built

For each incident the generator draws, in order: category (weighted) → sub­category
(uniform within the category) → service/region/host → signal values (sampled
within per-signal spec ranges) → title & description (templates filled with the
above) → severity (weighted per subcategory) → `customer_impacting` → priority →
channel (weighted per category) → `reported_at` offset.

Priority is derived from severity (`SEV1→P1 … SEV4→P4`) and bumped up one level
when the incident is customer-impacting — a small, realistic correlation rather
than an independent random label.

### Content catalog

[`catalog.py`](../src/nirnayax/data/catalog.py) holds the tunable content, kept
separate from the generation logic:

- `CATEGORY_WEIGHTS` — NETWORK 0.30, APPLICATION_DB 0.35, BILLING_OSS 0.20,
  HARDWARE_ACCESS 0.15.
- `REGIONS` — 8 telecom DC/edge site codes.
- `CATEGORY_CHANNEL_WEIGHTS` — realistic intake mix per category
  (network incidents skew to `MONITORING`, access issues to `PHONE`/`SELF_SERVICE`).
- `SUBCATEGORY_TEMPLATES` — one `IncidentTemplate` per subcategory: titles,
  descriptions, signal specs, severity mix, tags, customer-impact rate. Title and
  description placeholders (`{service}`, `{region}`, `{host}`, and each signal
  name) are checked as resolvable by a structural test.

---

## IDs and splits

| Split | Records | ID range | Default seed |
| --- | --- | --- | --- |
| `train` | 600 | `INC-000001` … `INC-000600` | `20260901` |
| `eval`  | 200 | `INC-900001` … `INC-900200` | `77777` |

Train IDs start at `TRAIN_ID_START = 1`; eval IDs start at
`EVAL_ID_START = 900_001`. The disjoint ranges mean train and eval **never
collide**, even when merged into one corpus — while both are drawn from the same
distribution. `generate_incidents(..., id_start=...)` guards the 6-digit
`INC-######` space and rejects a non-positive start.

---

## Runbooks

[`build_runbooks()`](../src/nirnayax/data/runbooks.py) constructs **17 runbooks —
one per subcategory** — from a compact spec table, so every triage label has a
remediation guide. IDs are stable (`RB-<CATEGORY>-<NNN>`, numbered in taxonomy
order) and `related_runbook_ids` cross-references are resolved to real IDs in a
second pass. `last_reviewed` is pinned so the catalog is reproducible.

---

## Validation

Per-record invariants (types, ID format, taxonomy consistency, field bounds) are
already guaranteed by the Pydantic models. [`validation.py`](../src/nirnayax/data/validation.py)
adds **collection-level** checks a single record can't express, split into fatal
`errors` and non-fatal `warnings`:

| Function | Errors | Warnings |
| --- | --- | --- |
| `validate_incidents` | duplicate IDs; taxonomy mismatch; `reported_at` after the reference clock | subcategories absent from the dataset |
| `validate_dataset` | the above + `metadata.size` ≠ incident count | — |
| `validate_runbooks` | duplicate IDs; a subcategory with no runbook; unresolved `related_runbook_ids` | a runbook referencing itself |

`ValidationReport` exposes `.ok`, `.raise_for_status()`, and `.render()`.

---

## EDA

[`compute_eda()`](../src/nirnayax/data/eda.py) returns a typed `EdaReport`
(machine-readable and `.render()`-able) using only `collections` + `statistics`
— no pandas, keeping the core dependency-light. It reports counts by category /
subcategory / severity / priority / channel / region, customer-impacting share,
title & description length stats, and the `reported_at` span.

---

## Persistence

[`serialization.py`](../src/nirnayax/data/serialization.py) provides lossless
round-trips:

- `save_dataset` / `load_dataset` — canonical **JSON** (`IncidentDataset` with
  metadata). This is the committed artifact.
- `save_incidents_jsonl` / `load_incidents_jsonl` — one incident per line, a
  streaming-friendly mirror (git-ignored; regenerate on demand).
- `save_runbooks` / `load_runbooks` — the runbook catalog as JSON.

---

## CLI

Run via `nirnayax <cmd>` (installed entry point) or `python -m nirnayax <cmd>`:

```bash
nirnayax generate [--out data] [--train-size 600] [--eval-size 200] [--seed N]
nirnayax validate data/incidents_train.json
nirnayax eda data/incidents_eval.json
```

`generate` writes the runbooks + both splits (JSON and JSONL), validates each,
and prints an EDA summary. It exits non-zero if any validation fails.

---

## Current dataset statistics

Generated with the default seeds and `REFERENCE_TIME = 2026-09-01`.

### Train — 600 incidents

| Dimension | Distribution |
| --- | --- |
| Category | APPLICATION_DB 34.2% · NETWORK 32.3% · BILLING_OSS 20.0% · HARDWARE_ACCESS 13.5% |
| Severity | SEV2 42.2% · SEV3 38.3% · SEV1 12.5% · SEV4 7.0% |
| Priority | P2 36.8% · P1 34.2% · P3 25.2% · P4 3.8% |
| Channel  | MONITORING 65.8% · EMAIL 14.2% · PHONE 11.3% · CHAT 5.3% · SELF_SERVICE 3.3% |
| Customer-impacting | 49.8% |
| Description length | min 104 · mean 157.9 · median 162 · max 212 |
| `reported_at` span | 2026-06-03 → 2026-08-31 |

All 17 subcategories present.

### Eval — 200 incidents

| Dimension | Distribution |
| --- | --- |
| Category | NETWORK 35.0% · APPLICATION_DB 35.0% · BILLING_OSS 20.0% · HARDWARE_ACCESS 10.0% |
| Severity | SEV2 46.0% · SEV3 36.5% · SEV1 12.5% · SEV4 5.0% |
| Priority | P1 37.5% · P2 36.5% · P3 22.5% · P4 3.5% |
| Channel  | MONITORING 74.0% · EMAIL 12.5% · PHONE 5.5% · CHAT 4.5% · SELF_SERVICE 3.5% |
| Customer-impacting | 49.0% |
| Description length | min 104 · mean 157.4 · median 160 · max 213 |
| `reported_at` span | 2026-06-04 → 2026-08-30 |

All 17 subcategories present.

> Regenerate any time with `nirnayax generate`; output is stable across runs.
