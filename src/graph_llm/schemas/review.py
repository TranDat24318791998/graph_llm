"""Schemas of the graph review: generated cards and recorded review decisions."""

from enum import StrEnum

from pydantic import Field

from graph_llm.schemas.base import StrictModel


class ReviewFlag(StrictModel):
    """One automatic check that needs a human look."""

    code: str = Field(description="Short identifier of the check, for example S-MIXED.")
    message: str


class StreetCard(StrictModel):
    """The computed facts and flags of one regular street."""

    street_id: str
    name: str
    direction: str
    centre_line_length_m: float
    two_way_length_m: float
    divided_length_m: float
    one_way_length_m: float
    junction_ids: list[str]
    meeting_street_names: list[str]
    places: list[str]
    inner_dead_end_node_ids: list[int]
    touches_area_boundary: bool
    latitude: float
    longitude: float
    flags: list[ReviewFlag]


class JunctionCard(StrictModel):
    """The computed facts and flags of one junction where regular streets meet."""

    junction_id: str
    kind: str
    node_count: int
    spread_m: float
    place_names: list[str]
    regular_street_names: list[str]
    unconditional_car_restrictions: int
    conditional_car_restrictions: int
    unmapped_restrictions: int
    latitude: float
    longitude: float
    flags: list[ReviewFlag]


class ReviewDecisionKind(StrEnum):
    """Outcome of reviewing one card."""

    ACCEPTED = "accepted"
    ACCEPTED_WITH_OVERRIDE = "accepted_with_override"
    EXCLUDED_FROM_QUESTIONS = "excluded_from_questions"


class ReviewDecision(StrictModel):
    """The recorded review of one card (street or junction)."""

    item_id: str = Field(description="street_id or junction_id.")
    decision: ReviewDecisionKind
    flags_seen: list[str] = Field(description="Flag codes present when the card was reviewed.")
    reason: str


class ReviewLog(StrictModel):
    """All review decisions of one graph version."""

    graph_version: str
    reviewer: str
    decisions: list[ReviewDecision]
