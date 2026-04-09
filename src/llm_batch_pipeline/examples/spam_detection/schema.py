"""Pydantic schema for spam detection output.

Follows the ``mySchema`` convention used by schema_loader.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SpamDetectionResult(BaseModel):
    """Schema for LLM spam classification output."""

    model_config = ConfigDict(extra="forbid")

    classification: Literal["spam", "ham"] = Field(description='Classification label. Must be exactly "spam" or "ham".')
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    reason: str = Field(
        min_length=1,
        description="Brief explanation of why the email was classified as spam or ham.",
    )
    indicators: list[str] = Field(
        default_factory=list,
        description="List of concrete spam or legitimacy indicators observed.",
    )
    suspicious_urls: list[str] = Field(
        default_factory=list,
        description="Suspicious or spam-related URLs found in the email body.",
    )
    sender_analysis: str = Field(
        default="",
        description="Brief analysis of the sender address and sending infrastructure.",
    )


class mySchema(SpamDetectionResult):  # noqa: N801
    """Convention class loaded by schema_loader."""
