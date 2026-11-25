"""Schemas for experiment runs: manifest, per-question predictions and scores."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from graph_llm.schemas.base import StrictModel
from graph_llm.schemas.question import Answer


class Representation(StrEnum):
    """The ways the road network is presented to the LLM.

    M0-M4 are the five representations of the proposal. M3N and M3T are Phase 1
    variants (EXP-002): M3 with intersection names instead of node ids, and the M1
    text corpus selected through the street graph instead of by similarity.
    """

    M0_NO_RETRIEVAL = "M0"
    M1_TEXT_RAG = "M1"
    M2_LLM_BUILT_GRAPH = "M2"
    M3_SUBGRAPH = "M3"
    M3_SUBGRAPH_NAMED_NODES = "M3N"
    M3_GRAPH_LINKED_TEXT = "M3T"
    M4_GRAPH_TOOLS = "M4"


class ErrorStage(StrEnum):
    """The pipeline stage a wrong answer is attributed to."""

    NONE = "none"
    OUTPUT_FORMAT = "output_format"
    NO_CONTEXT = "no_context"
    ENTITY_LINKING = "entity_linking"
    RETRIEVAL = "retrieval"
    USAGE = "usage"
    GRAPH_CONSTRUCTION = "graph_construction"


class RunManifest(StrictModel):
    """Everything needed to identify and reproduce one run."""

    run_id: str
    experiment_id: str = Field(description="Id of the approved protocol, for example EXP-001.")
    representation: Representation
    model_name: str
    benchmark_version: str
    graph_version: str = Field(description="Version of the OSM snapshot the graph was built from.")
    git_commit: str
    seed: int
    config: dict[str, Any] = Field(description="Full resolved configuration of the run.")
    created_at: datetime


class Prediction(StrictModel):
    """The LLM output for one question in one run."""

    run_id: str
    question_id: str
    context: str = Field(description="Exact context inserted into the prompt. Empty for M0.")
    raw_response: str
    parsed_answer: Answer | None = Field(
        description="Answer parsed from the raw response, or None if parsing failed."
    )
    prompt_tokens: int
    completion_tokens: int
    llm_call_count: int = Field(
        ge=1,
        description="Number of LLM calls made for this question. Above 1 for tool use (M4).",
    )
    latency_s: float = Field(description="Wall-clock time for the whole question, in seconds.")


class Score(StrictModel):
    """The evaluation result for one prediction."""

    run_id: str
    question_id: str
    is_correct: bool
    score: float = Field(ge=0.0, le=1.0, description="Partial credit; 1.0 when fully correct.")
    evidence_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Share of required evidence present in the context. None for M0 and M4.",
    )
    error_stage: ErrorStage
    sub_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Type-specific components, for example route legality and optimality for L4.",
    )
