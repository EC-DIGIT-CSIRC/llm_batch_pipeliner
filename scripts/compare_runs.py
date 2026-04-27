#!/usr/bin/env python3
"""Compare two completed batch directories and emit comparison.json.

Usage:
    scripts/compare_runs.py <ollama_batch_dir> <vllm_batch_dir> [output_path]

The script reads:
- output/summary.json
- export/evaluation.json
- logs/pipeline.jsonl (row_summary event)
- results/validated.json (to compute label agreement on `classification` if both
  exist and have overlapping filenames)
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_row_summary(log_path: Path) -> dict[str, Any]:
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if '"status": "row_summary"' in line:
            return json.loads(line)
    raise FileNotFoundError(f"row_summary event not found in {log_path}")


def load_labels(validated_path: Path) -> dict[str, str]:
    rows = read_json(validated_path)
    labels: dict[str, str] = {}
    for row in rows:
        filename = row.get("filename")
        label = row.get("classification")
        if filename and label is not None:
            labels[str(filename)] = str(label)
    return labels


def build_backend_summary(batch_dir: Path) -> dict[str, Any]:
    summary = read_json(batch_dir / "output" / "summary.json")
    evaluation = read_json(batch_dir / "export" / "evaluation.json")
    row = read_row_summary(batch_dir / "logs" / "pipeline.jsonl")

    return {
        "backend": summary.get("backend"),
        "service_run_key": _extract_run_key(batch_dir),
        "requests_per_second": summary.get("requests_per_second"),
        "completed_requests": summary.get("completed_requests"),
        "failed_requests": summary.get("failed_requests"),
        "row_duration_avg_ms": row.get("row_duration_avg_ms"),
        "row_duration_p50_ms": row.get("row_duration_p50_ms"),
        "row_duration_min_ms": row.get("row_duration_min_ms"),
        "row_duration_max_ms": row.get("row_duration_max_ms"),
        "accuracy": evaluation.get("accuracy"),
        "macro_precision": evaluation.get("macro_precision"),
        "macro_recall": evaluation.get("macro_recall"),
        "macro_f1": evaluation.get("macro_f1"),
        "weighted_precision": evaluation.get("weighted_precision"),
        "weighted_recall": evaluation.get("weighted_recall"),
        "weighted_f1": evaluation.get("weighted_f1"),
        "total_predictions": evaluation.get("total_predictions"),
        "per_class": evaluation.get("per_class"),
        "confusion_matrix": evaluation.get("confusion_matrix"),
    }


def label_agreement(labels_a: dict[str, str], labels_b: dict[str, str]) -> dict[str, Any]:
    common = sorted(set(labels_a) & set(labels_b))
    if not common:
        return {"rate": None, "compared": 0}
    same = sum(1 for key in common if labels_a[key] == labels_b[key])
    return {"rate": round(same / len(common), 4), "compared": len(common)}


def triple_agreement(labels_a: dict[str, str], labels_b: dict[str, str], labels_c: dict[str, str]) -> dict[str, Any]:
    common = sorted(set(labels_a) & set(labels_b) & set(labels_c))
    if not common:
        return {"all_same_rate": None, "majority_rate": None, "compared": 0}

    all_same = 0
    majority = 0
    for key in common:
        values = (labels_a[key], labels_b[key], labels_c[key])
        if values[0] == values[1] == values[2]:
            all_same += 1
            majority += 1
        elif values[0] == values[1] or values[0] == values[2] or values[1] == values[2]:
            majority += 1

    compared = len(common)
    return {
        "all_same_rate": round(all_same / compared, 4),
        "majority_rate": round(majority / compared, 4),
        "compared": compared,
    }


def main(argv: list[str]) -> int:
    if len(argv) not in {3, 4, 5}:
        print(
            "Usage: compare_runs.py <backend_a_batch_dir> <backend_b_batch_dir> [backend_c_batch_dir] [output_path]",
            file=sys.stderr,
        )
        return 2

    batch_dirs: list[Path]
    out_path: Path
    if len(argv) == 3:
        batch_dirs = [Path(argv[1]), Path(argv[2])]
        out_path = Path("comparison.json")
    elif len(argv) == 4:
        third = Path(argv[3])
        if third.is_dir():
            batch_dirs = [Path(argv[1]), Path(argv[2]), third]
            out_path = Path("comparison.json")
        else:
            batch_dirs = [Path(argv[1]), Path(argv[2])]
            out_path = third
    else:
        batch_dirs = [Path(argv[1]), Path(argv[2]), Path(argv[3])]
        out_path = Path(argv[4])

    backend_summaries: dict[str, dict[str, Any]] = {}
    label_maps: dict[str, dict[str, str]] = {}
    sample_size = None
    model = None
    for batch_dir in batch_dirs:
        summary = read_json(batch_dir / "output" / "summary.json")
        backend = str(summary.get("backend") or batch_dir.name)
        backend_summary = build_backend_summary(batch_dir)
        backend_summaries[backend] = backend_summary
        label_maps[backend] = _load_labels_if_present(batch_dir)
        total_requests = int(summary.get("total_requests", 0) or 0)
        sample_size = total_requests if sample_size is None else min(sample_size, total_requests)
        model = model or summary.get("model")

    result: dict[str, Any] = {
        "model": model,
        "sample_size": sample_size,
    }

    result.update(backend_summaries)

    backend_names = list(backend_summaries)
    if len(backend_names) == 2:
        labels_a = label_maps[backend_names[0]]
        labels_b = label_maps[backend_names[1]]
        agreement = label_agreement(labels_a, labels_b)
        result["label_agreement_rate"] = agreement["rate"]
        result["label_agreement_compared"] = agreement["compared"]
    else:
        pairwise: dict[str, Any] = {}
        for left, right in combinations(backend_names, 2):
            agreement = label_agreement(label_maps[left], label_maps[right])
            pairwise[f"{left}_vs_{right}"] = agreement
        result["pairwise_label_agreement"] = pairwise
        result["triple_label_agreement"] = triple_agreement(
            label_maps[backend_names[0]],
            label_maps[backend_names[1]],
            label_maps[backend_names[2]],
        )

    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


def _load_labels_if_present(batch_dir: Path) -> dict[str, str]:
    try:
        return load_labels(batch_dir / "results" / "validated.json")
    except FileNotFoundError:
        return {}


def _extract_run_key(batch_dir: Path) -> str | None:
    for line in (batch_dir / "logs" / "pipeline.jsonl").read_text(encoding="utf-8").splitlines():
        if '"service_run_key":' in line:
            obj = json.loads(line)
            return obj.get("service_run_key")
    return None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
