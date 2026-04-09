"""Test fixtures: sample .eml files and helper factories."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

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
