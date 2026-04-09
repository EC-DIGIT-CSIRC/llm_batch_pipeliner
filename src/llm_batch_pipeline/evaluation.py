"""Evaluation: confusion matrix, precision / recall / F1, accuracy, ROC.

Compares validated LLM predictions against ground-truth labels loaded from
a CSV file or inferred from filename prefixes via a category map.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_batch_pipeline.logging_utils import log_event

logger = logging.getLogger("llm_batch_pipeline.evaluation")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PredictionRow:
    custom_id: str
    ground_truth: str
    predicted: str
    confidence: float | None = None
    raw_output: dict[str, Any] | None = None


@dataclass(slots=True)
class ClassMetrics:
    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class EvalReport:
    """Complete evaluation report."""

    rows: list[PredictionRow] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    confusion: list[list[int]] = field(default_factory=list)
    per_class: list[ClassMetrics] = field(default_factory=list)
    accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_precision: float = 0.0
    weighted_recall: float = 0.0
    weighted_f1: float = 0.0
    roc_data: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_precision": round(self.weighted_precision, 4),
            "weighted_recall": round(self.weighted_recall, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "per_class": [c.to_dict() for c in self.per_class],
            "confusion_matrix": {
                "labels": self.labels,
                "matrix": self.confusion,
            },
            "total_predictions": len(self.rows),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalReport:
        """Reconstruct an EvalReport from serialised JSON (without rows)."""
        cm = data.get("confusion_matrix", {})
        labels = cm.get("labels", [])
        confusion = cm.get("matrix", [])
        per_class_raw = data.get("per_class", [])
        per_class = [
            ClassMetrics(
                label=c["label"],
                tp=c.get("tp", 0),
                fp=c.get("fp", 0),
                fn=c.get("fn", 0),
            )
            for c in per_class_raw
        ]
        return cls(
            labels=labels,
            confusion=confusion,
            per_class=per_class,
            accuracy=data.get("accuracy", 0.0),
            macro_precision=data.get("macro_precision", 0.0),
            macro_recall=data.get("macro_recall", 0.0),
            macro_f1=data.get("macro_f1", 0.0),
            weighted_precision=data.get("weighted_precision", 0.0),
            weighted_recall=data.get("weighted_recall", 0.0),
            weighted_f1=data.get("weighted_f1", 0.0),
        )


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------


def load_ground_truth_csv(path: Path) -> dict[str, str]:
    """Load ``{filename: label}`` from a two-column CSV.

    Auto-detects whether the first row is a header.
    """
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    if not rows:
        return mapping

    # Header detection: if both cells are purely alphabetic/underscore/hyphen
    first = rows[0]
    if len(first) >= 2 and all(c.replace("_", "").replace("-", "").replace(" ", "").isalpha() for c in first[:2]):
        rows = rows[1:]

    for row in rows:
        if len(row) >= 2:
            filename = row[0].strip()
            label = row[1].strip().lower()
            if filename:
                mapping[filename] = label

    return mapping


def infer_ground_truth_from_prefix(
    custom_id: str,
    category_map: dict[str, str],
) -> str | None:
    """Infer ground-truth label from filename prefix via category map.

    Splits on ``__`` (double underscore) and looks up the prefix.
    """
    if "__" in custom_id:
        prefix = custom_id.split("__", 1)[0]
        normalized = prefix.strip().lower()
        for key, value in category_map.items():
            if key.lower() == normalized:
                return value
    return None


def load_category_map(path: Path) -> dict[str, str]:
    """Load a ``{prefix: label}`` JSON mapping."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Evaluation computation
# ---------------------------------------------------------------------------


def evaluate(
    predictions: list[PredictionRow],
    *,
    positive_class: str | None = None,
) -> EvalReport:
    """Compute evaluation metrics from prediction rows."""
    if not predictions:
        return EvalReport()

    # Discover labels
    label_set: set[str] = set()
    for row in predictions:
        label_set.add(row.ground_truth)
        label_set.add(row.predicted)
    labels = sorted(label_set)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n = len(labels)

    # Build confusion matrix
    confusion = [[0] * n for _ in range(n)]
    for row in predictions:
        gt_idx = label_to_idx.get(row.ground_truth)
        pr_idx = label_to_idx.get(row.predicted)
        if gt_idx is not None and pr_idx is not None:
            confusion[gt_idx][pr_idx] += 1

    # Per-class metrics
    per_class: list[ClassMetrics] = []
    for i, label in enumerate(labels):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(n)) - tp
        fn = sum(confusion[i][j] for j in range(n)) - tp
        per_class.append(ClassMetrics(label=label, tp=tp, fp=fp, fn=fn))

    # Accuracy
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(n))
    accuracy = correct / total if total > 0 else 0.0

    # Macro averages
    macro_p = sum(c.precision for c in per_class) / n if n > 0 else 0.0
    macro_r = sum(c.recall for c in per_class) / n if n > 0 else 0.0
    macro_f1 = sum(c.f1 for c in per_class) / n if n > 0 else 0.0

    # Weighted averages
    total_support = sum(c.support for c in per_class)
    if total_support > 0:
        weighted_p = sum(c.precision * c.support for c in per_class) / total_support
        weighted_r = sum(c.recall * c.support for c in per_class) / total_support
        weighted_f1 = sum(c.f1 * c.support for c in per_class) / total_support
    else:
        weighted_p = weighted_r = weighted_f1 = 0.0

    # ROC data (binary classification with confidence scores)
    roc_data: list[dict[str, Any]] = []
    if positive_class and len(labels) == 2:
        roc_data = _compute_roc(predictions, positive_class)

    report = EvalReport(
        rows=predictions,
        labels=labels,
        confusion=confusion,
        per_class=per_class,
        accuracy=accuracy,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        weighted_precision=weighted_p,
        weighted_recall=weighted_r,
        weighted_f1=weighted_f1,
        roc_data=roc_data,
    )

    log_event(
        logger,
        f"Evaluation complete: accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}",
        step="evaluate",
        status="ok",
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        total=len(predictions),
        labels=labels,
    )
    return report


# ---------------------------------------------------------------------------
# ROC computation
# ---------------------------------------------------------------------------


def _compute_roc(
    predictions: list[PredictionRow],
    positive_class: str,
) -> list[dict[str, Any]]:
    """Compute ROC curve data points for binary classification."""
    scored = [(row.confidence, row.ground_truth == positive_class) for row in predictions if row.confidence is not None]
    if not scored:
        return []

    # Sort by confidence descending
    scored.sort(key=lambda x: x[0], reverse=True)

    total_pos = sum(1 for _, is_pos in scored if is_pos)
    total_neg = len(scored) - total_pos
    if total_pos == 0 or total_neg == 0:
        return []

    points: list[dict[str, Any]] = [{"fpr": 0.0, "tpr": 0.0, "threshold": 1.0}]
    tp = 0
    fp = 0

    for confidence, is_positive in scored:
        if is_positive:
            tp += 1
        else:
            fp += 1
        fpr = fp / total_neg
        tpr = tp / total_pos
        points.append({"fpr": round(fpr, 6), "tpr": round(tpr, 6), "threshold": round(confidence, 6)})

    return points


def compute_auc(roc_points: list[dict[str, Any]]) -> float:
    """Compute AUC via the trapezoidal rule from ROC data points."""
    if len(roc_points) < 2:
        return 0.0
    auc = 0.0
    for i in range(1, len(roc_points)):
        dx = roc_points[i]["fpr"] - roc_points[i - 1]["fpr"]
        avg_y = (roc_points[i]["tpr"] + roc_points[i - 1]["tpr"]) / 2
        auc += dx * avg_y
    return round(auc, 6)
