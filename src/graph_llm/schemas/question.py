"""Schemas for benchmark questions, their answers and their required evidence."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from graph_llm.schemas.base import StrictModel


class QuestionType(StrEnum):
    """The five question types of the benchmark, ordered by difficulty."""

    L1_ATTRIBUTE = "L1"
    L2_ADJACENCY = "L2"
    L3_MULTI_STEP = "L3"
    L4_ROUTE = "L4"
    L5_AGGREGATION = "L5"


class EdgeRef(StrictModel):
    """Reference to one directed edge of the ground-truth OSM graph.

    OSMnx stores the road network as a MultiDiGraph, so an edge is identified
    by its source node, target node and a key that separates parallel edges.
    """

    source_node_id: int
    target_node_id: int
    key: int = 0


class Evidence(StrictModel):
    """The nodes and edges of the ground-truth graph needed to answer a question."""

    node_ids: list[int] = Field(default_factory=list)
    edges: list[EdgeRef] = Field(default_factory=list)


class EntityLinkingChallenge(StrEnum):
    """A reason why a street named in a question is hard to match to the graph."""

    DUPLICATE_NAME = "duplicate_name"
    ABBREVIATED_NAME = "abbreviated_name"


class QuestionFeatures(StrictModel):
    """Structural properties of a question, computed by code when it is generated.

    They are recorded for every question so that any RQ4 method direction can be
    developed later without regenerating the benchmark.
    """

    hop_count: int = Field(
        ge=0,
        description="Number of edge traversals needed to reach the answer from the named entities.",
    )
    requires_computation: bool = Field(
        description="True if the answer needs arithmetic, counting or path search over the graph."
    )
    depends_on_edge_direction: bool = Field(
        description="True if reversing the direction of an evidence edge would change the answer."
    )
    entity_linking_challenges: list[EntityLinkingChallenge] = Field(default_factory=list)


class BooleanAnswer(StrictModel):
    """Yes/no answer, for example whether a street is one-way."""

    kind: Literal["boolean"] = "boolean"
    value: bool


class StreetNameAnswer(StrictModel):
    """A single street name."""

    kind: Literal["street_name"] = "street_name"
    street_name: str


class StreetSetAnswer(StrictModel):
    """An unordered set of street names, scored with set F1."""

    kind: Literal["street_set"] = "street_set"
    street_names: list[str]


class NumberAnswer(StrictModel):
    """A numeric answer such as a count."""

    kind: Literal["number"] = "number"
    value: float


class RouteAnswer(StrictModel):
    """A route as an ordered street sequence with its total length."""

    kind: Literal["route"] = "route"
    street_names: list[str]
    length_m: float
    node_path: list[int] = Field(
        default_factory=list,
        description="Node ids along the route. Set for ground-truth answers only.",
    )


Answer = Annotated[
    BooleanAnswer | StreetNameAnswer | StreetSetAnswer | NumberAnswer | RouteAnswer,
    Field(discriminator="kind"),
]


class Question(StrictModel):
    """One benchmark question with its computed answer and required evidence."""

    question_id: str
    benchmark_version: str
    question_type: QuestionType
    template_id: str
    template_text: str = Field(description="Question as produced by the template.")
    text: str = Field(description="Paraphrased question that is shown to the LLM.")
    slots: dict[str, str] = Field(
        default_factory=dict,
        description="Template slot values, for example the sampled street names.",
    )
    answer: Answer = Field(description="Ground truth, always computed from the graph by code.")
    evidence: Evidence
    features: QuestionFeatures
