"""Phase 2 ML triage engine: TF-IDF + Logistic Regression baseline.

Predicts **category**, **subcategory**, and **priority** for an incident ticket,
each with a calibrated confidence, from a single shared TF-IDF representation.

Requires the optional ``ml`` extra (``pip install -e ".[ml]"``); importing this
package pulls in scikit-learn. The Phase 1 data foundation stays dependency-light
and does not import from here.

Typical use::

    from nirnayax.data import load_dataset
    from nirnayax.ml import train_triage_model, evaluate_model, TicketDraft

    model = train_triage_model(load_dataset("data/incidents_train.json"))
    report = evaluate_model(model, load_dataset("data/incidents_eval.json"))
    prediction = model.predict(TicketDraft(title="...", description="..."))
"""

from __future__ import annotations

from .evaluation import evaluate_model, expected_calibration_error
from .features import TicketDraft, text_from_draft, text_from_incident
from .model import TriageModel
from .training import MODEL_VERSION, train_triage_model
from .types import (
    TARGETS,
    ClassMetrics,
    ClassPrediction,
    EvaluationReport,
    FeatureConfig,
    ModelMetadata,
    TargetMetrics,
    TrainingConfig,
    TriagePrediction,
)

__all__ = [
    "MODEL_VERSION",
    "TARGETS",
    "ClassMetrics",
    "ClassPrediction",
    "EvaluationReport",
    "FeatureConfig",
    "ModelMetadata",
    "TargetMetrics",
    "TicketDraft",
    "TrainingConfig",
    "TriageModel",
    "TriagePrediction",
    "evaluate_model",
    "expected_calibration_error",
    "text_from_draft",
    "text_from_incident",
    "train_triage_model",
]
