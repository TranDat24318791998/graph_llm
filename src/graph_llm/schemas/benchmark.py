"""Schemas for the benchmark configuration and the stored benchmark manifest."""

from datetime import datetime

from pydantic import Field

from graph_llm.schemas.base import StrictModel


class QuestionTemplate(StrictModel):
    """One question template and how many questions to draw from it."""

    template_id: str
    text: str = Field(description="Template text with {slot} placeholders.")
    count: int = Field(ge=1)
    required_phrases: list[str] = Field(
        description="Phrases a paraphrase must keep, so that no condition of the question is lost."
    )


class BenchmarkConfig(StrictModel):
    """Everything that defines how a benchmark version is generated."""

    benchmark_version: str
    graph_version: str
    split: str = Field(description="Which split this set belongs to, for example 'development'.")
    seed: int
    templates: list[QuestionTemplate]
    paraphrase_instruction: str = Field(description="Instruction given to the paraphrasing LLM.")


class BenchmarkManifest(StrictModel):
    """Provenance of a stored benchmark version."""

    config: BenchmarkConfig
    created_at: datetime
    question_count: int
    counts_by_template: dict[str, int]
    paraphrase_model: str
    paraphrase_fallback_question_ids: list[str] = Field(
        description=(
            "Questions whose paraphrase failed the automatic check; they keep the template text."
        )
    )
    questions_sha256: str
