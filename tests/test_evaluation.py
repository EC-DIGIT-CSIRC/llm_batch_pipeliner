"""Tests for evaluation.py — confusion matrix, metrics, ground truth."""

from llm_batch_pipeline.evaluation import (
    ClassMetrics,
    EvalReport,
    PredictionRow,
    compute_auc,
    evaluate,
    infer_ground_truth_from_prefix,
    load_category_map,
    load_ground_truth_csv,
)


def _pred(custom_id: str, gt: str, pred: str, conf: float | None = None) -> PredictionRow:
    return PredictionRow(custom_id=custom_id, ground_truth=gt, predicted=pred, confidence=conf)


class TestEvaluate:
    def test_perfect_binary(self):
        predictions = [
            _pred("a", "spam", "spam"),
            _pred("b", "ham", "ham"),
            _pred("c", "spam", "spam"),
            _pred("d", "ham", "ham"),
        ]
        report = evaluate(predictions)
        assert report.accuracy == 1.0
        assert report.macro_f1 == 1.0

    def test_all_wrong(self):
        predictions = [
            _pred("a", "spam", "ham"),
            _pred("b", "ham", "spam"),
        ]
        report = evaluate(predictions)
        assert report.accuracy == 0.0

    def test_confusion_matrix_shape(self):
        predictions = [
            _pred("a", "spam", "spam"),
            _pred("b", "ham", "ham"),
            _pred("c", "spam", "ham"),
        ]
        report = evaluate(predictions)
        assert len(report.labels) == 2
        assert len(report.confusion) == 2
        assert all(len(row) == 2 for row in report.confusion)

    def test_per_class_metrics(self):
        predictions = [
            _pred("a", "spam", "spam"),
            _pred("b", "spam", "spam"),
            _pred("c", "spam", "ham"),  # FN for spam, FP for ham
            _pred("d", "ham", "ham"),
        ]
        report = evaluate(predictions)
        spam_metrics = next(c for c in report.per_class if c.label == "spam")
        assert spam_metrics.tp == 2
        assert spam_metrics.fn == 1
        assert spam_metrics.recall == 2 / 3

    def test_empty_predictions(self):
        report = evaluate([])
        assert report.accuracy == 0.0
        assert report.labels == []

    def test_roc_data_with_positive_class(self):
        predictions = [
            _pred("a", "spam", "spam", 0.9),
            _pred("b", "ham", "ham", 0.2),
            _pred("c", "spam", "spam", 0.8),
            _pred("d", "ham", "spam", 0.6),
        ]
        report = evaluate(predictions, positive_class="spam")
        assert len(report.roc_data) > 0


class TestEvalReportSerialization:
    def test_to_dict_and_from_dict(self):
        predictions = [
            _pred("a", "spam", "spam"),
            _pred("b", "ham", "ham"),
        ]
        report = evaluate(predictions)
        d = report.to_dict()
        assert d["accuracy"] == 1.0
        assert "confusion_matrix" in d
        assert "per_class" in d

        # Round-trip
        restored = EvalReport.from_dict(d)
        assert restored.accuracy == 1.0
        assert restored.labels == report.labels
        assert restored.confusion == report.confusion


class TestClassMetrics:
    def test_perfect_class(self):
        cm = ClassMetrics(label="spam", tp=10, fp=0, fn=0)
        assert cm.precision == 1.0
        assert cm.recall == 1.0
        assert cm.f1 == 1.0
        assert cm.support == 10

    def test_zero_division(self):
        cm = ClassMetrics(label="empty", tp=0, fp=0, fn=0)
        assert cm.precision == 0.0
        assert cm.recall == 0.0
        assert cm.f1 == 0.0


class TestGroundTruth:
    def test_load_csv_with_header(self, tmp_path):
        csv_path = tmp_path / "gt.csv"
        csv_path.write_text("filename,label\na.eml,spam\nb.eml,ham\n", encoding="utf-8")
        result = load_ground_truth_csv(csv_path)
        assert result == {"a.eml": "spam", "b.eml": "ham"}

    def test_load_csv_without_header(self, tmp_path):
        csv_path = tmp_path / "gt.csv"
        csv_path.write_text("file1.eml,spam\nfile2.eml,ham\n", encoding="utf-8")
        result = load_ground_truth_csv(csv_path)
        # file1 starts with a digit + lowercase = detected as data, not header
        assert len(result) == 2

    def test_infer_from_prefix(self):
        cat_map = {"spam": "spam", "ham": "ham"}
        assert infer_ground_truth_from_prefix("spam__001.eml", cat_map) == "spam"
        assert infer_ground_truth_from_prefix("ham__002.eml", cat_map) == "ham"
        assert infer_ground_truth_from_prefix("unknown.eml", cat_map) is None

    def test_load_category_map(self, tmp_path):
        import json

        path = tmp_path / "catmap.json"
        path.write_text(json.dumps({"spam_": "spam", "ham_": "ham"}), encoding="utf-8")
        result = load_category_map(path)
        assert result == {"spam_": "spam", "ham_": "ham"}


class TestComputeAuc:
    def test_perfect_roc(self):
        points = [
            {"fpr": 0.0, "tpr": 0.0, "threshold": 1.0},
            {"fpr": 0.0, "tpr": 1.0, "threshold": 0.5},
            {"fpr": 1.0, "tpr": 1.0, "threshold": 0.0},
        ]
        auc = compute_auc(points)
        assert auc == 1.0

    def test_diagonal_roc(self):
        points = [
            {"fpr": 0.0, "tpr": 0.0, "threshold": 1.0},
            {"fpr": 1.0, "tpr": 1.0, "threshold": 0.0},
        ]
        auc = compute_auc(points)
        assert auc == 0.5

    def test_empty_roc(self):
        assert compute_auc([]) == 0.0
