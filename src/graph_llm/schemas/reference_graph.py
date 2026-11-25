"""Schemas of the graph v2 reference graph: its config and its frozen metadata."""

from datetime import datetime

from pydantic import Field

from graph_llm.schemas.base import StrictModel
from graph_llm.schemas.graph import SnapshotMetadata
from graph_llm.schemas.junction import JunctionSettings
from graph_llm.schemas.restriction import RestrictionSnapshotMetadata
from graph_llm.schemas.street import GraphProcessingConfig, NameResolution


class ReferenceGraphConfig(StrictModel):
    """Everything that defines one version of the reference graph (graph v2)."""

    streets: GraphProcessingConfig = Field(
        description="Street layer rules: graph_version, source snapshot, merging, overrides, "
        "aliases, internal roads (as in graph v1)."
    )
    junctions: JunctionSettings = Field(default_factory=JunctionSettings)
    review_log_path: str | None = Field(
        default=None, description="YAML file with the review decisions of this version."
    )


class ReferenceGraphMetadata(StrictModel):
    """Provenance and counts of one frozen reference graph."""

    config: ReferenceGraphConfig
    source_snapshot: SnapshotMetadata
    source_restrictions: RestrictionSnapshotMetadata
    created_at: datetime
    osmnx_version: str
    snapshot_node_count: int
    snapshot_edge_count: int
    simplified_node_count: int
    simplified_edge_count: int
    node_count: int
    edge_count: int
    removed_node_ids: list[int] = Field(
        description="Nodes outside the largest strongly connected component."
    )
    removed_edge_count: int
    unnamed_edge_count: int
    street_count: int
    place_count: int
    junction_count: int
    name_resolutions: list[NameResolution]
    graphml_sha256: str
    streets_sha256: str
    junctions_sha256: str
