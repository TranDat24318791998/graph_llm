"""Shared base class for all schemas."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Immutable model that rejects unknown fields.

    Records are written to disk and read back months later. Rejecting unknown
    fields makes a schema mismatch fail loudly instead of silently dropping data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
