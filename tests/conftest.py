"""Test fixtures: sample .eml files and helper factories."""

from __future__ import annotations

import json
import shutil
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Sample email bodies
# ---------------------------------------------------------------------------

SIMPLE_HAM_EML = textwrap.dedent("""\
    From: alice@example.com
    To: bob@example.com
    Subject: Meeting tomorrow
    Date: Mon, 1 Jan 2024 10:00:00 +0000
    MIME-Version: 1.0
    Content-Type: text/plain; charset="utf-8"

    Hi Bob,

    Can we meet tomorrow at 3pm to discuss the project?

    Thanks,
    Alice
""")

SIMPLE_SPAM_EML = textwrap.dedent("""\
    From: deals@cheap-pills.biz
    To: victim@example.com
    Subject: URGENT!! You have WON $1,000,000!!!
    Date: Mon, 1 Jan 2024 10:00:00 +0000
    MIME-Version: 1.0
    Content-Type: text/plain; charset="utf-8"

    Congratulations! You have been selected as the winner of our grand prize!
    Click here to claim your $1,000,000: http://scam.example.com/claim

    This is a limited time offer! Act NOW or lose your prize forever!
    Send your bank details to claim@cheap-pills.biz immediately.
""")

HTML_EMAIL = textwrap.dedent("""\
    From: newsletter@company.com
    To: user@example.com
    Subject: Weekly Newsletter
    Date: Mon, 1 Jan 2024 10:00:00 +0000
    MIME-Version: 1.0
    Content-Type: text/html; charset="utf-8"

    <html><body>
    <h1>Weekly Newsletter</h1>
    <p>Hello! Here is your weekly update.</p>
    <p>Visit our <a href="https://company.com">website</a> for more.</p>
    </body></html>
""")

MULTIPART_EML = textwrap.dedent("""\
    From: sender@example.com
    To: receiver@example.com
    Subject: Test multipart
    Date: Mon, 1 Jan 2024 10:00:00 +0000
    MIME-Version: 1.0
    Content-Type: multipart/alternative; boundary="boundary123"

    --boundary123
    Content-Type: text/plain; charset="utf-8"

    Plain text version of the email.

    --boundary123
    Content-Type: text/html; charset="utf-8"

    <html><body><p>HTML version of the email.</p></body></html>

    --boundary123--
""")

EMPTY_BODY_EML = textwrap.dedent("""\
    From: nobody@example.com
    To: someone@example.com
    Subject: Empty email
    Date: Mon, 1 Jan 2024 10:00:00 +0000
    MIME-Version: 1.0
    Content-Type: text/plain; charset="utf-8"

""")

PII_EMAIL = textwrap.dedent("""\
    From: hr@company.com
    To: manager@company.com
    Subject: Employee record - John Smith
    Date: Mon, 1 Jan 2024 10:00:00 +0000
    MIME-Version: 1.0
    Content-Type: text/plain; charset="utf-8"

    Please find the updated employee details:

    Name: John Smith
    SSN: 123-45-6789
    Date of Birth: 1985-03-15
    Address: 123 Main Street, Springfield, IL 62701
    Phone: +1 (555) 123-4567
    Email: john.smith@personal.com
    Bank Account: 1234567890 (routing: 021000021)
""")

AUTO_REPLY_EML = textwrap.dedent("""\
    From: vacation@example.com
    To: sender@example.com
    Subject: Automatic Reply: Out of Office
    Date: Mon, 1 Jan 2024 10:00:00 +0000
    MIME-Version: 1.0
    Content-Type: text/plain; charset="utf-8"

    I am currently out of the office with limited access to email.
    I will respond to your message when I return on January 15th.
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_eml(directory: Path, filename: str, content: str) -> Path:
    """Write an .eml file to a directory and return the path."""
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


def make_batch_output_record(
    custom_id: str,
    output_text: str,
    *,
    status_code: int = 200,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one OpenAI-format batch output record."""
    record: dict[str, Any] = {"custom_id": custom_id}
    if error is not None:
        record["error"] = error
    else:
        record["response"] = {
            "status_code": status_code,
            "body": {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": output_text,
                            }
                        ],
                    }
                ]
            },
        }
    return record


def write_batch_output_jsonl(
    directory: Path,
    filename: str,
    records: list[dict[str, Any]],
) -> Path:
    """Write a batch output JSONL file."""
    path = directory / filename
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Shared fixture batches
# ---------------------------------------------------------------------------


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
BATCH_ROUNDTRIP_FIXTURE_DIR = FIXTURES_DIR / "batch_roundtrip"


@pytest.fixture(scope="session")
def batch_roundtrip_fixture_dir() -> Path:
    return BATCH_ROUNDTRIP_FIXTURE_DIR


@pytest.fixture()
def copy_batch_roundtrip_fixture(tmp_path: Path, batch_roundtrip_fixture_dir: Path) -> Path:
    dest = tmp_path / "batch_roundtrip"
    shutil.copytree(batch_roundtrip_fixture_dir, dest, dirs_exist_ok=True)
    return dest


# ---------------------------------------------------------------------------
# Fake Ollama server for e2e tests
# ---------------------------------------------------------------------------


class _FakeOllamaServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, _FakeOllamaRequestHandler)
        self.requests: list[dict[str, Any]] = []


class _FakeOllamaRequestHandler(BaseHTTPRequestHandler):
    server: _FakeOllamaServer

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        payload = json.loads(raw_body.decode("utf-8"))
        self.server.requests.append(payload)
        response = _build_fake_ollama_response(payload)
        body = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: Any) -> None:
        return


def _build_fake_ollama_response(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages", [])
    content = "\n".join(str(message.get("content", "")) for message in messages if isinstance(message, dict)).lower()

    if content.strip() in {"hi", "hello"}:
        message = "warmup ok"
    elif any(token in content for token in ("winner", "prize", "bank details", "urgent", "scam", "cheap-pills")):
        message = json.dumps(
            {
                "classification": "spam",
                "confidence": 0.99,
                "reason": "Obvious spam indicators were detected.",
                "indicators": ["urgent subject", "prize language", "money request"],
                "suspicious_urls": ["http://scam.example.com/claim"],
                "sender_analysis": "Suspicious bulk sender domain.",
            },
            ensure_ascii=False,
        )
    else:
        message = json.dumps(
            {
                "classification": "ham",
                "confidence": 0.96,
                "reason": "No spam indicators were detected.",
                "indicators": ["routine meeting request"],
                "suspicious_urls": [],
                "sender_analysis": "Normal personal sender.",
            },
            ensure_ascii=False,
        )

    return {
        "model": payload.get("model", "gemma4:latest"),
        "message": {"content": message},
        "prompt_eval_count": 128,
        "eval_count": 64,
        "total_duration": 1_000_000,
        "load_duration": 100_000,
        "prompt_eval_duration": 400_000,
        "eval_duration": 500_000,
    }


@pytest.fixture()
def fake_ollama_server() -> tuple[str, list[dict[str, Any]]]:
    server = _FakeOllamaServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    try:
        yield base_url, server.requests
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


# ---------------------------------------------------------------------------
# Automatic marker assignment
# ---------------------------------------------------------------------------


_INTEGRATION_TESTS = {"test_pipeline.py", "test_stages.py"}
_E2E_TESTS = {"test_cli_roundtrip.py"}
_BENCHMARK_TESTS = {"test_inference_loop.py"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(getattr(item, "path", item.fspath)))
        name = path.name

        if "e2e" in path.parts or name in _E2E_TESTS:
            item.add_marker(pytest.mark.e2e)
            continue

        if name in _BENCHMARK_TESTS:
            item.add_marker(pytest.mark.benchmark)
            continue

        if name in _INTEGRATION_TESTS or name.startswith("test_backends_"):
            item.add_marker(pytest.mark.integration)
            if name.startswith("test_backends_"):
                item.add_marker(pytest.mark.contract)
            continue

        item.add_marker(pytest.mark.unit)
