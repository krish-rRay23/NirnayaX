"""The trained triage model: one shared TF-IDF space, three classifier heads.

A single :class:`TriageModel` bundles the fitted ``TfidfVectorizer`` and the
three fitted ``LogisticRegression`` heads (category, subcategory, priority) plus
:class:`~nirnayax.ml.types.ModelMetadata` for versioning. Sharing one vectorizer
computes the text representation once per ticket and keeps the artifact compact.

Construct one with :func:`nirnayax.ml.training.train_triage_model`; this module
owns *inference* and *persistence*.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib

from ..domain.models import Incident
from ..domain.taxonomy import Subcategory, category_of
from .features import TicketDraft, text_from_draft, text_from_incident
from .types import TARGETS, ClassPrediction, ModelMetadata, TriagePrediction


def _class_prediction(classes: Sequence[str], proba_row: Sequence[float]) -> ClassPrediction:
    """Build a :class:`ClassPrediction` from aligned classes and probabilities."""

    pairs = sorted(
        ((str(c), float(p)) for c, p in zip(classes, proba_row, strict=True)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    label, confidence = pairs[0]
    return ClassPrediction(label=label, confidence=confidence, distribution=tuple(pairs))


def _taxonomy_consistent(subcategory_label: str, category_label: str) -> bool:
    """True if the predicted subcategory's parent category matches the prediction."""

    try:
        return category_of(Subcategory(subcategory_label)).value == category_label
    except ValueError:
        return False


class TriageModel:
    """A fitted triage model providing inference and persistence."""

    def __init__(
        self,
        *,
        vectorizer: Any,
        classifiers: dict[str, Any],
        metadata: ModelMetadata,
    ) -> None:
        missing = [t for t in TARGETS if t not in classifiers]
        if missing:
            raise ValueError(f"missing classifier head(s): {missing}")
        self._vectorizer = vectorizer
        self._classifiers = classifiers
        self.metadata = metadata

    @property
    def model_version(self) -> str:
        return self.metadata.model_version

    def classes_for(self, target: str) -> tuple[str, ...]:
        """Return the label space a head can predict."""

        return tuple(str(c) for c in self._classifiers[target].classes_)

    # -- inference -----------------------------------------------------------
    def _predict_text(self, text: str) -> TriagePrediction:
        matrix = self._vectorizer.transform([text])
        heads: dict[str, ClassPrediction] = {}
        for target in TARGETS:
            clf = self._classifiers[target]
            classes = [str(c) for c in clf.classes_]
            proba_row = clf.predict_proba(matrix)[0]
            heads[target] = _class_prediction(classes, proba_row)

        subcategory = heads["subcategory"]
        category = heads["category"]
        return TriagePrediction(
            category=category,
            subcategory=subcategory,
            priority=heads["priority"],
            taxonomy_consistent=_taxonomy_consistent(subcategory.label, category.label),
            model_version=self.metadata.model_version,
        )

    def predict(self, draft: TicketDraft) -> TriagePrediction:
        """Predict category, subcategory, and priority for an incoming ticket."""

        return self._predict_text(text_from_draft(draft, self.metadata.training_config.features))

    def predict_incident(self, incident: Incident) -> TriagePrediction:
        """Predict for an existing :class:`Incident` (ignores its ground-truth labels)."""

        return self._predict_text(
            text_from_incident(incident, self.metadata.training_config.features)
        )

    def score_incidents(self, incidents: Sequence[Incident]) -> dict[str, tuple[list[str], Any]]:
        """Batch-score incidents; returns ``target -> (classes, proba_matrix)``.

        Used by evaluation: transforms all incidents once and runs each head over
        the whole matrix, which is far cheaper than per-incident prediction.
        """

        config = self.metadata.training_config.features
        texts = [text_from_incident(inc, config) for inc in incidents]
        matrix = self._vectorizer.transform(texts)
        scores: dict[str, tuple[list[str], Any]] = {}
        for target in TARGETS:
            clf = self._classifiers[target]
            classes = [str(c) for c in clf.classes_]
            scores[target] = (classes, clf.predict_proba(matrix))
        return scores

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist the model to ``path`` (joblib) plus a human-readable ``.meta.json``.

        Returns the model path. The metadata sidecar is written for auditing and
        is not required to load the model.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata.model_dump(mode="json"),
            "vectorizer": self._vectorizer,
            "classifiers": self._classifiers,
        }
        joblib.dump(payload, path)
        meta_path = path.parent / (path.name + ".meta.json")
        meta_path.write_text(self.metadata.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> TriageModel:
        """Load a model saved by :meth:`save`.

        .. warning::
           Uses ``joblib``/pickle under the hood — only load model files you trust.
        """

        payload = joblib.load(Path(path))
        metadata = ModelMetadata.model_validate(payload["metadata"])
        return cls(
            vectorizer=payload["vectorizer"],
            classifiers=payload["classifiers"],
            metadata=metadata,
        )


__all__ = ["TriageModel"]
