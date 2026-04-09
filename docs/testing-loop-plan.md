# Testing Loop Plan

This plan turns the Peter Steinberger's blog post  of [shipping at inference speed](https://steipete.me/posts/2025/shipping-at-inference-speed) into this repo's actual seams: pure logic, stage wiring, CLI entry points, backend boundaries, and CI.

## Loop Model

1. Fast loop: unit tests for pure functions and serializers.
2. Wiring loop: stage orchestration and artifact propagation.
3. Contract loop: backend request/response behavior without real services.
4. E2E loop: installed CLI over a real temp batch directory.
5. Benchmark loop: tiny corpus, threshold checks, optional live model runs.

## File-By-File Plan

| File | Change | Why |
|---|---|---|
| `pyproject.toml` | Add pytest markers for `unit`, `integration`, `e2e`, `contract`, `benchmark`, `live_openai`, and `live_ollama`. Keep test discovery simple and explicit. | Lets CI split fast and slow loops without brittle `-k` filters. |
| `tests/conftest.py` | Add fixtures for a canonical temp batch tree, tiny email corpus, schema/prompt files, fake backend responses, and helper assertion utilities. | Removes repeated setup and makes the tests read like workflows. |
| `tests/fixtures/batch_roundtrip/` (new) | Add a tiny one-ham/one-spam corpus with `input/`, `evaluation/ground-truth.csv`, `prompt.txt`, and `schema.py`. | Gives integration and E2E tests a shared real-world fixture. |
| `tests/test_backends_common.py` (new) | Cover `load_and_validate_batch()`, prompt override handling, atomic writes, and `build_summary()`. | This is the shared contract for both backends. |
| `tests/test_stages.py` (new) | Test `stage_discover`, `stage_filter_1`, `stage_transform`, `stage_filter_2`, `stage_render`, `stage_review`, `stage_submit`, `stage_validate`, `stage_output_transform`, `stage_evaluate`, `stage_export`, plus `build_pipeline()`. | Proves the pipeline wiring, not just the isolated helpers. |
| `tests/test_pipeline.py` | Extend the existing orchestrator tests to assert `build_pipeline()` stage order, optional stages, `start_from`, and saved state shape. | Prevents regressions in resume/skip behavior. |
| `tests/test_cli.py` | Add parser and dispatch coverage for every subcommand, including `test-inference` once wired. | Keeps the CLI contract stable as the primary human and agent interface. |
| `tests/test_backends_openai.py` (new) | Use injected fake OpenAI clients to test submit, poll, resume, no-wait, output download, and prompt override. | Covers the network boundary without hitting the network. |
| `tests/test_backends_ollama.py` (new) | Use monkeypatched `httpx.Client` or a small transport seam to test request translation, sharding, retry behavior, schema sanitization, and deterministic output ordering. | Covers the local backend without needing a live Ollama server. |
| `src/llm_batch_pipeline/backends/ollama_backend.py` | If the test seam is awkward, add a tiny client factory or transport injection point. Keep the production path unchanged. | Makes the contract tests deterministic and cheap. |
| `tests/test_render.py`, `tests/test_validation.py`, `tests/test_evaluation.py`, `tests/test_export.py` | Upgrade a few assertions from file-exists checks to content-level checks on JSONL records, workbook sheet names/cells, ROC data, and schema-driven ordering. | Converts shallow smoke coverage into real behavioral checks. |
| `src/llm_batch_pipeline/cli.py` | Finish wiring the existing `test-inference` parser into `build_parser()` and `main()`. Keep the command thin. | Gives the repo a CLI-first benchmark/verification loop. |
| `src/llm_batch_pipeline/inference_loop.py` (new) | Implement the actual loop: discover tiny corpus, render, submit, validate, evaluate, compare thresholds, emit summary JSON, and support `--no-check`. | Encapsulates the benchmark loop so it is testable and reusable. |
| `tests/test_inference_loop.py` (new) | Test threshold logic, summary output, and failure modes with fake backends. | Protects the benchmark loop from becoming a hand-run script only. |
| `tests/e2e/test_cli_roundtrip.py` (new) | Run the installed CLI against the shared fixture batch and assert the on-disk artifacts from `init` through `export`. | Proves the user-facing path end to end. |
| `.github/workflows/tests.yml` (new) | Add PR jobs for unit, integration, and E2E; add scheduled/manual benchmark and live smoke jobs. | Turns the repo into a looped system, not a single pass/fail gate. |
| `.github/workflows/ruff.yml`, `.github/workflows/pylint.yml`, `.github/workflows/semgrep.yml` | No functional change unless new test files need path filters. | Existing static checks can stay as-is. |
| `docs/benchmark-run.md` | Update the benchmark story to reference the new `test-inference` loop and thresholds. | Makes the slow loop discoverable and repeatable. |
| `README.md` | Add a short testing section pointing to the new docs and workflow commands. | Keeps the main entry point obvious. |

## Suggested Implementation Order

1. Add markers and shared fixtures.
2. Add backend and stage contract tests.
3. Add the CLI roundtrip test.
4. Wire `test-inference` into a small runner module.
5. Add the GitHub Actions test workflow.
6. Update docs and README links.

## Acceptance Criteria

1. `uv run pytest -m unit` stays fast and deterministic.
2. `uv run pytest -m integration` proves the pipeline stages together without live services.
3. `uv run pytest -m e2e` runs the installed CLI over a real temp batch directory.
4. `uv run llm-batch-pipeline test-inference ...` returns a machine-readable summary and fails on threshold regressions.
5. Live model checks are opt-in, scheduled, and never required for a pull request.
