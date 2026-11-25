"""Schemas of an experiment configuration and of aggregated run metrics."""

from pydantic import Field

from graph_llm.schemas.base import StrictModel
from graph_llm.schemas.llm import LLMEndpointConfig
from graph_llm.schemas.run import Representation
from graph_llm.schemas.street import StreetCategory


class LightRAGSettings(StrictModel):
    """Retrieval settings of M2 (LightRAG 1.5.8 QueryParam fields)."""

    mode: str
    top_k: int
    chunk_top_k: int
    max_entity_tokens: int
    max_relation_tokens: int
    max_total_tokens: int


class ExperimentConfig(StrictModel):
    """Everything that defines the runs of one experiment."""

    experiment_id: str
    benchmark_version: str
    graph_version: str
    corpus_id: str
    lightrag_index_id: str
    answer_llm: LLMEndpointConfig
    prompt_config_path: str
    embedding_model: str
    context_budget_characters: int = Field(
        gt=0, description="Upper bound on the context length, the same for every representation."
    )
    m1_top_k: int
    m3_hops: int
    m3t_hops: int = Field(
        default=2, description="Street-level hops of M3T (graph-linked text), added in EXP-002."
    )
    lightrag: LightRAGSettings
    question_linker_categories: list[StreetCategory] | None = Field(
        default=None,
        description=(
            "Street categories that M3, M3N and M3T link in a question; None links every "
            "street (EXP-001). Added in EXP-002."
        ),
    )
    representations: list[Representation]


class GroupMetrics(StrictModel):
    """Aggregated results of one group of questions (a type, a template, or all)."""

    question_count: int
    accuracy: float = Field(description="Share of questions answered fully correctly.")
    mean_score: float = Field(description="Mean score; equals accuracy except for L2 (set F1).")
    mean_evidence_recall: float | None
    parse_failure_rate: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_llm_calls: float
    mean_latency_s: float
    error_stage_counts: dict[str, int]
    sub_score_means: dict[str, float]


class RunMetrics(StrictModel):
    """Aggregated results of one run."""

    run_id: str
    overall: GroupMetrics
    by_question_type: dict[str, GroupMetrics]
    by_template: dict[str, GroupMetrics]
