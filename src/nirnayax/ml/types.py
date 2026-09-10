"""Typed configuration, prediction, metadata, and metrics models for the ML engine.

Deliberately free of scikit-learn imports so the *data contracts* of the triage
engine can be constructed, serialized, and tested without the heavy ML stack.
The fitted estimators and the training/evaluation logic live in sibling modules.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

#: The prediction heads, in a stable order.
TARGETS: tuple[str, ...] = ("category", "subcategory", "urgency", "impact", "priority")


class _Frozen(BaseModel):
    """Immutable, strict base (mirrors the domain layer's conventions)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class FeatureConfig(_Frozen):
    """TF-IDF featurization settings (versioned into model metadata)."""

    ngram_max: int = Field(default=2, ge=1, le=3)
    min_df: int = Field(default=2, ge=1)
    max_df: float = Field(default=0.9, gt=0.0, le=1.0)
    max_features: int | None = Field(default=25000, ge=100)
    sublinear_tf: bool = True
    use_stopwords: bool = True
    include_metadata: bool = True


class TrainingConfig(_Frozen):
    """Reproducible training hyper-parameters for all heads."""

    seed: int = 20260901
    C: float = Field(default=4.0, gt=0.0)
    max_iter: int = Field(default=100, ge=1)
    #: ``None`` keeps predicted probabilities well calibrated; ``"balanced"``
    #: trades calibration for recall on rare classes.
    class_weight: str | None = None
    features: FeatureConfig = FeatureConfig()


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
class ClassPrediction(_Frozen):
    """A single head's output: the winning label, its confidence, full distribution."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    #: ``(label, probability)`` pairs, sorted by descending probability.
    distribution: tuple[tuple[str, float], ...]

    def top_k(self, k: int) -> tuple[tuple[str, float], ...]:
        """Return the ``k`` most probable ``(label, probability)`` pairs."""

        return self.distribution[:k]

    def render(self) -> str:
        return f"{self.label} ({self.confidence:.1%})"


class TriagePrediction(_Frozen):
    """The full triage result for one ticket."""

    category: ClassPrediction
    subcategory: ClassPrediction
    urgency: ClassPrediction | None = None
    impact: ClassPrediction | None = None
    priority: ClassPrediction
    #: True when the predicted subcategory's parent equals the predicted category.
    taxonomy_consistent: bool
    model_version: str

    def render(self) -> str:
        flag = "" if self.taxonomy_consistent else "  [!] subcategory/category mismatch"
        urg_str = f"\nurgency     : {self.urgency.render()}" if self.urgency else ""
        imp_str = f"\nimpact      : {self.impact.render()}" if self.impact else ""
        return (
            f"category    : {self.category.render()}\n"
            f"subcategory : {self.subcategory.render()}"
            f"{urg_str}"
            f"{imp_str}\n"
            f"priority    : {self.priority.render()}\n"
            f"consistent  : {self.taxonomy_consistent}{flag}"
        )


# ---------------------------------------------------------------------------
# Model metadata / versioning
# ---------------------------------------------------------------------------
class ModelMetadata(_Frozen):
    """Provenance stamped onto a trained model for versioning and auditing."""

    schema_version: str = "1"
    model_version: str
    created_at: datetime
    python_version: str
    sklearn_version: str
    numpy_version: str
    train_dataset_name: str
    train_dataset_split: str
    train_size: int = Field(ge=0)
    train_dataset_seed: int
    #: Deterministic fingerprint of the training texts + labels.
    train_fingerprint: str
    targets: tuple[str, ...]
    label_space: dict[str, tuple[str, ...]]
    training_config: TrainingConfig

    def render(self) -> str:
        lines = [
            f"model_version   : {self.model_version}",
            f"created_at      : {self.created_at.isoformat()}",
            f"trained_on      : {self.train_dataset_name} "
            f"(split={self.train_dataset_split}, n={self.train_size}, "
            f"seed={self.train_dataset_seed})",
            f"fingerprint     : {self.train_fingerprint}",
            f"sklearn/numpy   : {self.sklearn_version} / {self.numpy_version}",
            f"python          : {self.python_version}",
            "label space     : "
            + ", ".join(f"{t}={len(self.label_space[t])}" for t in self.targets),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
class ClassMetrics(_Frozen):
    """Per-class precision/recall/F1/support."""

    precision: float
    recall: float
    f1: float
    support: int


class TargetMetrics(_Frozen):
    """Evaluation metrics for one prediction head, including its confusion matrix."""

    target: str
    n: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    #: Expected Calibration Error of the top prediction's confidence (lower = better).
    ece: float
    mean_confidence: float
    labels: tuple[str, ...]
    #: Row = true label, column = predicted label (aligned with ``labels``).
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: dict[str, ClassMetrics]

    def render(self) -> str:
        lines = [
            f"[{self.target}] n={self.n}  acc={self.accuracy:.3f}  "
            f"macro-F1={self.macro_f1:.3f}  weighted-F1={self.weighted_f1:.3f}",
            f"           macro-P={self.macro_precision:.3f}  "
            f"macro-R={self.macro_recall:.3f}  "
            f"ECE={self.ece:.3f}  mean-conf={self.mean_confidence:.3f}",
            _render_confusion(self.labels, self.confusion_matrix),
        ]
        return "\n".join(lines)


class EvaluationReport(_Frozen):
    """Evaluation across all heads on a held-out dataset."""

    model_version: str
    dataset_name: str
    dataset_size: int
    targets: dict[str, TargetMetrics]

    def render(self) -> str:
        header = (
            f"Evaluation - model {self.model_version} on {self.dataset_name} "
            f"({self.dataset_size} incidents)"
        )
        blocks = [header, "=" * len(header)]
        for name in TARGETS:
            if name in self.targets:
                blocks.append(self.targets[name].render())
        return "\n\n".join(blocks)


def _render_confusion(labels: tuple[str, ...], matrix: tuple[tuple[int, ...], ...]) -> str:
    """Render a confusion matrix compactly: an index legend + a numeric grid."""

    width = max((len(str(cell)) for row in matrix for cell in row), default=1)
    width = max(width, 2)
    idx_w = len(str(len(labels) - 1)) if labels else 1

    legend = "  legend: " + "  ".join(f"{i}={lab}" for i, lab in enumerate(labels))
    header = " " * (idx_w + 3) + " ".join(f"{i:>{width}}" for i in range(len(labels)))
    rows = [
        f"  {i:>{idx_w}} | " + " ".join(f"{cell:>{width}}" for cell in row)
        for i, row in enumerate(matrix)
    ]
    return "\n".join(["  confusion (row=true, col=pred):", legend, header, *rows])


__all__ = [
    "TARGETS",
    "ClassMetrics",
    "ClassPrediction",
    "EvaluationReport",
    "FeatureConfig",
    "ModelMetadata",
    "TargetMetrics",
    "TrainingConfig",
    "TriagePrediction",
]
