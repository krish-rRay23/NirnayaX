"""Reproducible training pipeline: fit one TF-IDF space and three LogReg heads.

Training is deterministic given ``(dataset, TrainingConfig)``: the vectorizer and
the ``lbfgs`` solver are deterministic, and ``random_state`` is pinned. Only the
training split is ever fitted here — evaluation lives in :mod:`nirnayax.ml.evaluation`.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..domain.models import Incident, IncidentDataset
from .features import text_from_incident
from .model import TriageModel
from .types import TARGETS, ModelMetadata, TrainingConfig

#: Bump when the model architecture/feature contract changes (persisted in metadata).
MODEL_VERSION = "0.1.0"


def _fingerprint(texts: Sequence[str], labels: Mapping[str, Sequence[str]]) -> str:
    """Deterministic short digest of the exact training texts + labels."""

    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    for target in TARGETS:
        digest.update(target.encode("utf-8"))
        for label in labels[target]:
            digest.update(label.encode("utf-8"))
            digest.update(b"\x00")
    return digest.hexdigest()[:16]


def train_triage_model(
    dataset: IncidentDataset,
    config: TrainingConfig | None = None,
    *,
    model_version: str = MODEL_VERSION,
) -> TriageModel:
    """Fit a :class:`TriageModel` on ``dataset`` (the training split).

    Deterministic for a given ``(dataset, config)``.
    """

    config = config or TrainingConfig()
    incidents = dataset.incidents
    if not incidents:
        raise ValueError("cannot train on an empty dataset")

    features = config.features
    texts = [text_from_incident(inc, features) for inc in incidents]

    def _extract_tag_val(inc: Incident, prefix: str, default: str) -> str:
        for tag in inc.tags:
            if tag.startswith(prefix):
                return tag[len(prefix) :]
        return default

    labels: dict[str, list[str]] = {
        "category": [_extract_tag_val(inc, "raw_cat:", inc.category.value) for inc in incidents],
        "subcategory": [
            _extract_tag_val(inc, "raw_sub1:", inc.subcategory.value) for inc in incidents
        ],
        "urgency": [_extract_tag_val(inc, "urgency_", "3") for inc in incidents],
        "impact": [_extract_tag_val(inc, "impact_", "4") for inc in incidents],
        "priority": [inc.priority.value for inc in incidents],
    }

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, features.ngram_max),
        min_df=features.min_df,
        max_df=features.max_df,
        max_features=features.max_features,
        sublinear_tf=features.sublinear_tf,
        stop_words="english" if features.use_stopwords else None,
    )
    matrix = vectorizer.fit_transform(texts)

    from sklearn.dummy import DummyClassifier

    classifiers: dict[str, Any] = {}
    label_space: dict[str, tuple[str, ...]] = {}
    for target in TARGETS:
        unique_labels = set(labels[target])
        if len(unique_labels) < 2:
            clf: Any = DummyClassifier(strategy="most_frequent")
        else:
            clf = LogisticRegression(
                C=config.C,
                max_iter=config.max_iter,
                class_weight=config.class_weight,
                solver="lbfgs",
                random_state=config.seed,
            )
        clf.fit(matrix, labels[target])
        classifiers[target] = clf
        label_space[target] = tuple(str(c) for c in clf.classes_)

    metadata = ModelMetadata(
        model_version=model_version,
        created_at=datetime.now(UTC),
        python_version=platform.python_version(),
        sklearn_version=str(sklearn.__version__),
        numpy_version=str(numpy.__version__),
        train_dataset_name=dataset.metadata.name,
        train_dataset_split=dataset.metadata.split.value,
        train_size=len(incidents),
        train_dataset_seed=dataset.metadata.seed,
        train_fingerprint=_fingerprint(texts, labels),
        targets=TARGETS,
        label_space=label_space,
        training_config=config,
    )
    return TriageModel(vectorizer=vectorizer, classifiers=classifiers, metadata=metadata)


__all__ = ["MODEL_VERSION", "train_triage_model"]
