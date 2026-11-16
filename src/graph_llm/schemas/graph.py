"""Schemas for the study area, the frozen OSM snapshot and its attribute coverage."""

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from graph_llm.schemas.base import StrictModel


class BoundingBox(StrictModel):
    """A rectangle in WGS84 degrees."""

    west: float = Field(ge=-180.0, le=180.0)
    south: float = Field(ge=-90.0, le=90.0)
    east: float = Field(ge=-180.0, le=180.0)
    north: float = Field(ge=-90.0, le=90.0)

    @model_validator(mode="after")
    def check_corner_order(self) -> Self:
        """Reject a rectangle whose corners are swapped."""
        if self.west >= self.east:
            raise ValueError("west must be smaller than east")
        if self.south >= self.north:
            raise ValueError("south must be smaller than north")
        return self


class StudyArea(StrictModel):
    """The geographic area the road graph is downloaded for."""

    area_id: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    bounding_box: BoundingBox
    network_type: str = Field(description="OSMnx network type, for example 'drive'.")
    simplify: bool = Field(
        description="If true, OSMnx removes nodes that are not intersections or dead ends."
    )


class SnapshotMetadata(StrictModel):
    """Provenance of one frozen OSM download."""

    area: StudyArea
    downloaded_at: datetime
    osmnx_version: str
    node_count: int
    edge_count: int
    graphml_sha256: str = Field(description="Checksum of the stored GraphML file.")


class AttributeCoverage(StrictModel):
    """How much of the graph carries a value for one edge attribute."""

    attribute: str
    edges_with_value: int
    edge_share: float = Field(ge=0.0, le=1.0, description="Share of edges that have a value.")
    length_share: float = Field(
        ge=0.0, le=1.0, description="Share of total edge length that has a value."
    )
    multi_valued_edges: int = Field(
        description="Edges whose value is a list, because simplification merged differing segments."
    )


class CoverageReport(StrictModel):
    """Attribute coverage and basic statistics of one graph snapshot."""

    area_id: str
    node_count: int
    edge_count: int
    total_length_m: float
    oneway_edge_count: int
    distinct_street_name_count: int
    attributes: list[AttributeCoverage]
    highway_class_edge_counts: dict[str, int]
