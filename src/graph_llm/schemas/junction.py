"""Schemas of the junction layer of the processed graph (graph v2)."""

from enum import StrEnum

from pydantic import Field

from graph_llm.schemas.base import StrictModel


class JunctionKind(StrEnum):
    """What a junction is made of."""

    SINGLE_NODE = "single_node"
    COMPACT = "compact"
    DIVIDED_CROSSING = "divided_crossing"
    WITH_CONNECTORS = "with_connectors"
    ROUNDABOUT_OR_PLACE = "roundabout_or_place"


class JunctionSettings(StrictModel):
    """Thresholds and name rules of the junction layer (see docs/graph_v2_design.md)."""

    link_max_m: float = Field(
        default=80.0, description="Longest unnamed connecting edge inside a junction."
    )
    median_crossing_max_m: float = Field(
        default=40.0, description="Longest edge that crosses the median of a divided road."
    )
    tiny_edge_max_m: float = Field(
        default=12.0, description="Edges up to this length between intersections are internal."
    )
    carriageway_max_separation_m: float = Field(
        default=40.0, description="Largest distance between the two carriageways of one street."
    )
    split_crossing_link_max_m: float = Field(
        default=60.0,
        description="Longest edge joining two node groups of one crossing that are merged.",
    )
    street_like_places: list[str] = Field(
        default_factory=list,
        description="Names with a place prefix that are streets (reviewed decision with reason).",
    )
    place_name_prefixes: list[str] = Field(
        default=[
            "Công trường",
            "Quảng trường",
            "Ngã ba",
            "Ngã tư",
            "Ngã năm",
            "Ngã sáu",
            "Ngã bảy",
            "Vòng xoay",
        ],
        description="Names starting with these words name a square or roundabout, not a street.",
    )


class Junction(StrictModel):
    """One real intersection: one or more road-layer nodes."""

    junction_id: str = Field(description="'j' followed by the smallest node id.")
    node_ids: list[int]
    latitude: float
    longitude: float
    spread_m: float = Field(description="Largest distance between two of its nodes.")
    kind: JunctionKind
    internal_edge_reasons: list[str]
    place_names: list[str] = Field(description="Names of squares or roundabouts it forms.")
    street_ids: list[str] = Field(description="Streets (not places) that reach the junction.")
