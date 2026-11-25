"""Schemas for the processed road graph and its street layer.

The processed graph has two layers:

- the road layer is the OSM graph itself, reduced to its largest strongly
  connected component; no edge is added or changed, so routing stays exact;
- the street layer groups road edges into named streets; this is where pieces of
  one street that OSM stores apart are joined.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from graph_llm.schemas.base import StrictModel
from graph_llm.schemas.graph import SnapshotMetadata
from graph_llm.schemas.question import EdgeRef


class StreetCategory(StrEnum):
    """What kind of named way a street is. Only regular streets are asked about in Phase 1."""

    REGULAR = "regular"
    ALLEY = "alley"
    INTERNAL = "internal"


class OverrideDecision(StrEnum):
    """A manual decision for a name whose pieces the automatic rule leaves apart."""

    MERGE = "merge"
    SEPARATE = "separate"


class StreetOverride(StrictModel):
    """A reviewed decision about one street name, with its reason."""

    name: str = Field(description="The name as spelled in OSM.")
    decision: OverrideDecision
    reason: str


class NameAliases(StrictModel):
    """Other ways people write one street name, for example abbreviations."""

    name: str = Field(description="The name as spelled in OSM.")
    aliases: list[str] = Field(min_length=1)


class GraphProcessingConfig(StrictModel):
    """Everything that defines how a frozen snapshot becomes a processed graph."""

    graph_version: str = Field(pattern=r"^[a-z0-9_]+$")
    source_area_id: str
    merge_gap_m: float = Field(
        gt=0.0,
        description=(
            "Pieces of the same name are one street if a path of at most this length, "
            "ignoring edge direction, connects them."
        ),
    )
    internal_road_names: list[str] = Field(
        description="Names of roads inside residential compounds, which are not public streets."
    )
    street_overrides: list[StreetOverride]
    name_aliases: list[NameAliases]


class NameResolution(StrictModel):
    """How the pieces of one name that OSM stores apart were turned into streets."""

    name_key: str
    piece_count: int = Field(ge=2, description="Connected pieces of the name in the road graph.")
    groups_after_automatic_rule: int = Field(ge=1)
    override: OverrideDecision | None
    street_ids: list[str]


class Street(StrictModel):
    """One named street: a set of directed edges of the road graph."""

    street_id: str
    display_name: str
    name_key: str
    ascii_key: str
    category: StreetCategory
    osm_names: list[str] = Field(description="Every OSM spelling that was mapped to this street.")
    aliases: list[str]
    shares_name_with_other_street: bool = Field(
        description="True if a different street carries the same name, so the name is ambiguous."
    )
    edges: list[EdgeRef]


class StreetCatalog(StrictModel):
    """All streets of one processed graph."""

    graph_version: str
    streets: list[Street]


class ProcessedGraphMetadata(StrictModel):
    """Provenance of one processed graph."""

    config: GraphProcessingConfig
    source_snapshot: SnapshotMetadata
    created_at: datetime
    osmnx_version: str
    node_count: int
    edge_count: int
    removed_node_ids: list[int] = Field(
        description="Nodes outside the largest strongly connected component."
    )
    removed_edge_count: int
    unnamed_edge_count: int
    street_count: int
    name_resolutions: list[NameResolution]
    graphml_sha256: str
    streets_sha256: str
