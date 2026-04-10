# Testing Loop Details

This document records the preferred Ollama testing strategy for later implementation.

## Core Idea

Use real `ollama` on `localhost:11434` once to capture a known-good batch run, then replay that capture in tests.

The goal is to keep the test loop grounded in real model behavior while making the day-to-day test suite deterministic and fast.

## Why This Is Better Than A Handwritten Fake Server

1. The first capture comes from a real model and real backend behavior.
2. The replay fixture becomes a stable oracle for CI.
3. Request drift is detected explicitly instead of being hidden by a loose mock.
4. The replay fixture can include both success and failure cases.
5. The same capture can support CLI, backend, validation, and evaluation checks.

## Recommended Flow

### 1. Capture

Run the spam dataset against real Ollama and record:

1. The rendered request JSONL.
2. The raw backend responses.
3. The validated rows.
4. The evaluation report.
5. The model name and backend settings used.

### 2. Package

Store the capture as a fixture bundle:

```text
tests/fixtures/ollama_cassettes/spam_roundtrip/
  manifest.json
  requests.jsonl
  responses.jsonl
  expected/
    validated.json
    evaluation.json
    summary.json
```

### 3. Replay

Replay tests should read the fixture bundle and return recorded answers for matching requests.

### 4. Diff

Run validation and evaluation against the replayed responses and compare the outputs to the recorded expectations.

## Request Matching

Do not match only on raw request text.

Use a normalized fingerprint built from:

1. `custom_id`
2. model
3. prompt or instructions hash
4. schema hash
5. packaged input hash
6. backend options that affect output

Ignore volatile fields such as timestamps, request ids, and result ordering noise.

## Replay Server Behavior

The replay layer should be data-driven.

1. Verify that every incoming request has a matching cassette entry.
2. Return the recorded response for that entry.
3. Fail immediately on unexpected, missing, or duplicate requests.
4. Treat the warmup call as either a recorded special case or a dedicated fixture entry.
5. Keep strict mode as the default for CI.

## Comparison Strategy

Prefer two comparison modes:

1. Strict: exact request matching and exact expected outputs.
2. Semantic: allow known harmless drift such as rounding differences.

For this repo, strict should be the default.

## What To Record

The best fixture bundle should include:

1. The model used during capture.
2. The exact prompts and schema used.
3. The output JSONL from the backend.
4. The final validated rows.
5. The evaluation JSON.
6. Any backend summary or timing metadata worth tracking.

## Failure Coverage

Also capture cases that test resilience:

1. One retry then success.
2. Final failure after retries.
3. Warmup succeeds.
4. Warmup fails.
5. Output mismatch.

## CI Split

Use two modes in CI:

1. PR tests replay the cassette only.
2. Nightly or manual jobs refresh the cassette from real Ollama.

That keeps pull requests deterministic while still allowing real-model refreshes.

## Refresh Policy

Refreshing the cassette should be an explicit act, not an accidental side effect.

1. Capture from real Ollama.
2. Review the diff in requests, responses, and evaluation.
3. Update the fixture only when the change is intentional.

## Summary

Preferred path:

1. Real Ollama for capture.
2. Recorded cassette for tests.
3. Strict replay for CI.
4. Explicit refresh for model drift.
