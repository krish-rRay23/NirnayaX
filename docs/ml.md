# ML triage engine

The ML layer turns the data foundation into a working triage model: given an
incoming ticket it predicts **category**, **subcategory**, and **priority**, each
with a **calibrated confidence**. It is a deliberately small, production-oriented
baseline — TF-IDF + Logistic Regression, no heavy frameworks.

- [`features`](../src/nirnayax/ml/features.py) — leakage-free text featurization
- [`training`](../src/nirnayax/ml/training.py) — reproducible fitting pipeline
- [`model`](../src/nirnayax/ml/model.py) — inference + versioned persistence
- [`evaluation`](../src/nirnayax/ml/evaluation.py) — metrics, confusion matrices, calibration
- [`types`](../src/nirnayax/ml/types.py) — typed config / prediction / metadata / metrics

The engine is an **optional extra** (`pip install -e ".[ml]"`). The core domain and
data layers never import it, so they stay dependency-light (pydantic only); the CLI
imports scikit-learn lazily, inside the ML subcommands.

---

## Architecture

One shared **`TfidfVectorizer`** produces a single sparse representation per ticket,
which feeds **three independent `LogisticRegression` heads** (category, subcategory,
priority):

```
ticket → build_text → TF-IDF (shared) → ┬→ LogReg(category)     → ClassPrediction
                                        ├→ LogReg(subcategory)  → ClassPrediction
                                        └→ LogReg(priority)     → ClassPrediction
```

Sharing the vectorizer computes the text representation once per ticket, keeps the
persisted artifact compact, and means the three heads always see identical features.
Heads are independent (not chained) so an error in one cannot cascade into another;
consistency between category and subcategory is *reported* (see below), not enforced.

---

## Feature contract & leakage avoidance

[`build_text`](../src/nirnayax/ml/features.py) is the **single** place raw fields
become model input, so training and inference cannot drift apart. It uses only
fields available **at intake time**:

- **Prose** — `title` and `description`, kept verbatim.
- **Categorical metadata** — `affected_service`, `region`, `channel`, and `tags`,
  emitted as namespaced tokens (`svc_…`, `region_…`, `chan_…`, `tag_…`) so the
  vectorizer treats them as discrete features rather than free text.

It deliberately **excludes `severity`, `priority`, `customer_impacting`, and
`status`**. Priority is *derived* from severity + customer impact during generation,
so feeding any of those back in would be label leakage. The inference input type
[`TicketDraft`](../src/nirnayax/ml/features.py) does not even define those fields
(`extra="forbid"`), so leakage is impossible by construction — a property the tests
assert directly.

---

## Determinism & reproducibility

Training is **deterministic** given `(dataset, TrainingConfig)`:

- The `TfidfVectorizer` and the `lbfgs` solver are deterministic, and
  `random_state` is pinned from `TrainingConfig.seed`.
- A **training fingerprint** — a SHA-256 over the exact training texts and labels —
  is stamped into the model metadata, so two artifacts can be compared for
  provenance at a glance.

Consequences (asserted by the test suite): the same `(dataset, config)` yields an
identical fingerprint, identical predictions, and — through evaluation — an
identical `EvaluationReport`, confusion matrices included.

Only the **training split** is ever fitted. Evaluation lives in a separate module and
only ever *scores* an already-trained model, and the train/eval datasets use disjoint
ID ranges and different seeds (see [docs/data.md](data.md)).

---

## Prediction output

Each head returns a [`ClassPrediction`](../src/nirnayax/ml/types.py): the winning
`label`, its `confidence` in `[0, 1]`, and the **full probability distribution** over
that head's label space (sorted, so `top_k` is trivial). The combined
[`TriagePrediction`](../src/nirnayax/ml/types.py) bundles all three heads plus a
`taxonomy_consistent` flag — `True` when the predicted subcategory's parent category
equals the predicted category. The flag surfaces disagreement between the two heads
rather than silently overriding one.

```
category    : NETWORK (91.7%)
subcategory : BGP_ROUTING (84.6%)
priority    : P2 (48.9%)
consistent  : True
```

---

## Evaluation & calibration

[`evaluate_model`](../src/nirnayax/ml/evaluation.py) reports, per head: accuracy,
macro/weighted F1, macro precision/recall, a per-class breakdown, a confusion matrix,
and the **Expected Calibration Error (ECE)** of the confidence output. ECE bins
predictions by confidence and averages `|accuracy − confidence|` across bins — it
answers "when the model says 80%, is it right ~80% of the time?"

`class_weight` defaults to `None` precisely because balancing distorts predicted
probabilities; keeping it off preserves calibration, which is the point of reporting
confidence at all.

### Baseline results

Model `0.1.0`, trained on 600 incidents, evaluated on the 200-incident held-out split:

| Head | Accuracy | Macro-F1 | Weighted-F1 | ECE | Mean confidence |
| --- | --- | --- | --- | --- | --- |
| category | 1.000 | 1.000 | 1.000 | 0.052 | 0.948 |
| subcategory | 1.000 | 1.000 | 1.000 | 0.109 | 0.891 |
| priority | 0.445 | 0.376 | 0.439 | 0.159 | 0.604 |

**Category / subcategory** are near-perfect: the synthetic templates use distinct
vocabulary per subcategory, so the classes are highly separable and their confusion
matrices are effectively diagonal. This is a property of the synthetic corpus, not
leakage — the featurizer excludes every label-derived field.

**Priority is intentionally hard.** It is derived from severity + customer impact,
which are *not* in the features, so text alone cannot fully determine it. The model
recovers the subcategory-conditioned base rate — beating the majority-class baseline
but capping well short of the easy heads. The priority confusion matrix concentrates
mass on the P1–P2–P3 band with the expected off-diagonal spread:

```
priority (row=true, col=pred):   0=P1  1=P2  2=P3  3=P4
   0 | 39 30  6  0
   1 | 22 37 14  0
   2 | 11 21 12  1
   3 |  0  2  4  1
```

This is the honest ceiling for a text-only priority model on this data, and it is
exactly why the per-head calibrated confidence is worth surfacing: downstream logic
can trust a 95%-confident category far more than a 49%-confident priority.

---

## Persistence & versioning

[`TriageModel.save`](../src/nirnayax/ml/model.py) writes two files:

- `triage.joblib` — the fitted vectorizer + heads + metadata (via `joblib`).
- `triage.joblib.meta.json` — a human-readable provenance sidecar: model version,
  timestamps, Python/scikit-learn/numpy versions, the training fingerprint, and the
  label space. It is for auditing and is not required to load the model.

`MODEL_VERSION` (in [`training`](../src/nirnayax/ml/training.py)) is bumped when the
architecture or feature contract changes. Loading uses `joblib`/pickle, so — as the
docstring warns — **only load model files you trust**.

---

## CLI

```bash
nirnayax train                 # fit on data/incidents_train.json -> models/triage.joblib
nirnayax evaluate              # score models/triage.joblib on data/incidents_eval.json
nirnayax predict --title "…" --description "…" [--service …] [--region …] [--channel …] [--tag …]
```

`train` accepts `--seed`, `--reg-c` (inverse regularization `C`), and `--balanced`
(trades calibration for rare-class recall). All paths are overridable; see
`nirnayax <cmd> --help`.

---

## Intentionally out of scope (later phases)

RAG / retrieval over the runbook knowledge base, LLM or agentic triage, ticketing/MCP
integration, remediation, and any UI. The baseline here is the measurable, testable
foundation those build on.
