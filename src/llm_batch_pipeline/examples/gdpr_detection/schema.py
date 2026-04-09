"""Pydantic schema for GDPR / sensitive data detection output.

Follows the ``mySchema`` convention used by schema_loader.
"""

from pydantic import BaseModel, ConfigDict, Field


class PiiCategory(BaseModel):
    """A single detected PII category with examples."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(
        description="PII category (e.g. 'email_address', 'phone_number', 'name', 'address', 'financial', 'health')."
    )
    count: int = Field(
        ge=0,
        description="Number of instances detected.",
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Up to 3 representative examples (redacted if needed).",
    )
    gdpr_article: str = Field(
        default="",
        description="Relevant GDPR article (e.g. 'Art. 9 - Special categories').",
    )


class GdprDetectionResult(BaseModel):
    """Schema for LLM GDPR/PII detection output."""

    model_config = ConfigDict(extra="forbid")

    contains_pii: bool = Field(description="Whether the email contains any personally identifiable information.")
    sensitivity_level: str = Field(description="Overall sensitivity: 'none', 'low', 'medium', 'high', 'critical'.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    pii_categories: list[PiiCategory] = Field(
        default_factory=list,
        description="List of detected PII categories.",
    )
    reason: str = Field(
        min_length=1,
        description="Brief explanation of the assessment.",
    )
    recommended_action: str = Field(
        default="",
        description="Recommended GDPR compliance action (e.g. 'redact', 'encrypt', 'delete', 'none').",
    )


class mySchema(GdprDetectionResult):  # noqa: N801
    """Convention class loaded by schema_loader."""
