"""Tests for export.py — XLSX export."""


from llm_batch_pipeline.evaluation import PredictionRow, evaluate
from llm_batch_pipeline.export import export_evaluation_xlsx, export_results_xlsx


class TestExportResultsXlsx:
    def test_basic_export(self, tmp_path):
        rows = [
            {"filename": "a.eml", "label": "spam", "confidence": 0.9},
            {"filename": "b.eml", "label": "ham", "confidence": 0.8},
        ]
        output = tmp_path / "results.xlsx"
        export_results_xlsx(rows, output)
        assert output.is_file()
        assert output.stat().st_size > 0

    def test_empty_rows(self, tmp_path):
        output = tmp_path / "results.xlsx"
        export_results_xlsx([], output)
        assert output.is_file()

    def test_with_schema_ordering(self, tmp_path):
        schema_code = (
            "from pydantic import BaseModel, Field\n\n"
            "class mySchema(BaseModel):\n"
            '    label: str = Field(description="x")\n'
            "    confidence: float = Field(default=0.0)\n"
        )
        schema_path = tmp_path / "schema.py"
        schema_path.write_text(schema_code, encoding="utf-8")

        rows = [
            {"filename": "a.eml", "label": "spam", "confidence": 0.9, "extra": "data"},
        ]
        output = tmp_path / "results.xlsx"
        export_results_xlsx(rows, output, schema_path=schema_path)
        assert output.is_file()


class TestExportEvaluationXlsx:
    def test_basic_export(self, tmp_path):
        predictions = [
            PredictionRow(custom_id="a", ground_truth="spam", predicted="spam", confidence=0.9),
            PredictionRow(custom_id="b", ground_truth="ham", predicted="ham", confidence=0.8),
            PredictionRow(custom_id="c", ground_truth="spam", predicted="ham", confidence=0.3),
        ]
        report = evaluate(predictions)
        output = tmp_path / "evaluation.xlsx"
        export_evaluation_xlsx(report, output)
        assert output.is_file()
        assert output.stat().st_size > 0

    def test_with_roc_data(self, tmp_path):
        predictions = [
            PredictionRow(custom_id="a", ground_truth="spam", predicted="spam", confidence=0.9),
            PredictionRow(custom_id="b", ground_truth="ham", predicted="ham", confidence=0.2),
        ]
        report = evaluate(predictions, positive_class="spam")
        output = tmp_path / "evaluation.xlsx"
        export_evaluation_xlsx(report, output)
        assert output.is_file()
