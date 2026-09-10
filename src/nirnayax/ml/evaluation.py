"""Evaluation on a held-out split: accuracy, F1, confusion matrices, calibration.

Never fits anything — it only scores an already-trained :class:`TriageModel` on a
dataset it has not seen. For each head it reports accuracy, macro/weighted F1,
macro precision/recall, a per-class breakdown, a confusion matrix, and the
**Expected Calibration Error (ECE)** of the confidence output.
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import classification_report, confusion_matrix

from ..domain.models import Incident, IncidentDataset
from .model import TriageModel
from .types import ClassMetrics, EvaluationReport, TargetMetrics


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> float:
    """Expected Calibration Error of top-label confidences (equal-width bins)."""

    n = len(confidences)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        members = [
            i
            for i in range(n)
            if (confidences[i] > lo or (b == 0 and confidences[i] >= lo)) and confidences[i] <= hi
        ]
        if not members:
            continue
        bin_acc = sum(1 for i in members if correct[i]) / len(members)
        bin_conf = sum(confidences[i] for i in members) / len(members)
        ece += (len(members) / n) * abs(bin_acc - bin_conf)
    return ece


def _target_metrics(
    target: str, y_true: list[str], classes: list[str], proba: Any
) -> TargetMetrics:
    n = len(y_true)
    pred_idx = [int(j) for j in proba.argmax(axis=1)]
    y_pred = [classes[j] for j in pred_idx]
    confidences = [float(proba[i, pred_idx[i]]) for i in range(n)]
    correct = [yp == yt for yp, yt in zip(y_pred, y_true, strict=True)]

    labels = tuple(sorted(classes))
    report = classification_report(
        y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))

    per_class = {
        label: ClassMetrics(
            precision=float(report[label]["precision"]),
            recall=float(report[label]["recall"]),
            f1=float(report[label]["f1-score"]),
            support=int(report[label]["support"]),
        )
        for label in labels
    }

    return TargetMetrics(
        target=target,
        n=n,
        accuracy=sum(1 for c in correct if c) / n,
        macro_f1=float(report["macro avg"]["f1-score"]),
        weighted_f1=float(report["weighted avg"]["f1-score"]),
        macro_precision=float(report["macro avg"]["precision"]),
        macro_recall=float(report["macro avg"]["recall"]),
        ece=expected_calibration_error(confidences, correct),
        mean_confidence=sum(confidences) / n,
        labels=labels,
        confusion_matrix=tuple(tuple(int(cell) for cell in row) for row in matrix),
        per_class=per_class,
    )


def evaluate_model(model: TriageModel, dataset: IncidentDataset) -> EvaluationReport:
    """Evaluate ``model`` on ``dataset`` across all three heads."""

    incidents = dataset.incidents
    if not incidents:
        raise ValueError("cannot evaluate on an empty dataset")

    scores = model.score_incidents(incidents)

    def _extract_tag_val(inc: Incident, prefix: str, default: str) -> str:
        for tag in inc.tags:
            if tag.startswith(prefix):
                return tag[len(prefix) :]
        return default

    truth: dict[str, list[str]] = {
        "category": [_extract_tag_val(inc, "raw_cat:", inc.category.value) for inc in incidents],
        "subcategory": [
            _extract_tag_val(inc, "raw_sub1:", inc.subcategory.value) for inc in incidents
        ],
        "urgency": [_extract_tag_val(inc, "urgency_", "3") for inc in incidents],
        "impact": [_extract_tag_val(inc, "impact_", "4") for inc in incidents],
        "priority": [inc.priority.value for inc in incidents],
    }

    targets = {
        target: _target_metrics(target, truth[target], *scores[target])
        for target in scores
        if target in truth
    }
    return EvaluationReport(
        model_version=model.model_version,
        dataset_name=dataset.metadata.name,
        dataset_size=len(incidents),
        targets=targets,
    )


__all__ = ["evaluate_model", "expected_calibration_error"]
