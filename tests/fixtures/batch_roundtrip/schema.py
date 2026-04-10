"""Schema fixture for the roundtrip batch."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SpamDetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["spam", "ham"] = Field(description="Classification label.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score.")
    reason: str = Field(min_length=1, description="Brief explanation.")
    indicators: list[str] = Field(default_factory=list, description="Observed indicators.")
    suspicious_urls: list[str] = Field(default_factory=list, description="Suspicious URLs.")
    sender_analysis: str = Field(default="", description="Sender analysis.")


class mySchema(SpamDetectionResult):
    pass
