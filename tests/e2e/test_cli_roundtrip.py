"""End-to-end CLI roundtrip against a fake local Ollama server."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from llm_batch_pipeline.cli import main


def test_cli_roundtrip_with_fake_ollama(
    tmp_path: Path,
    batch_roundtrip_fixture_dir: Path,
    fake_ollama_server: tuple[str, list[dict[str, object]]],
) -> None:
    base_url, requests = fake_ollama_server

    batch_jobs_root = tmp_path / "batches"
    rc = main(["init", "roundtrip", "--plugin", "spam_detection", "--batch-jobs-root", str(batch_jobs_root)])
    assert rc == 0

    batch_dir = batch_jobs_root / "batch_001_roundtrip"
    shutil.copytree(batch_roundtrip_fixture_dir, batch_dir, dirs_exist_ok=True)

    rc = main(
        [
            "run",
            "--batch-dir",
            str(batch_dir),
            "--plugin",
            "spam_detection",
            "--backend",
            "ollama",
            "--base-url",
            base_url,
            "--model",
            "gemma4:latest",
            "--prompt-file",
            str(batch_dir / "prompt.txt"),
            "--schema-file",
            str(batch_dir / "schema.py"),
            "--auto-approve",
            "--log-level",
            "ERROR",
        ]
    )
    assert rc == 0

    assert len(requests) >= 3
    assert requests[0]["messages"][0]["content"] == "hi"

    output_dir = batch_dir / "output"
    results_dir = batch_dir / "results"
    export_dir = batch_dir / "export"

    assert (output_dir / "output.jsonl").is_file()
    assert (results_dir / "validated.json").is_file()
    assert (export_dir / "evaluation.xlsx").is_file()
    assert (export_dir / "results.xlsx").is_file()
    assert (export_dir / "evaluation.json").is_file()

    evaluation = json.loads((export_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["accuracy"] == 1.0
    assert evaluation["macro_f1"] == 1.0
