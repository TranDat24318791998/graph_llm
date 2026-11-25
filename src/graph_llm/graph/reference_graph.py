"""Build, freeze and load the reference graph (graph v2).

Steps (docs/graph_v2_design.md):

1. Load an unsimplified snapshot and the turn restrictions frozen with it.
2. Simplify without merging different roads, keeping restriction nodes
   (``graph/simplify.py``).
3. Keep the largest strongly connected component (as in graph v1).
4. Build the street layer (v1 rules: names, merging, overrides, categories).
5. Classify carriageways and build junctions (``graph/carriageways.py``,
   ``graph/junctions.py``).

The result is written once to ``data/processed/<graph_version>/`` with checksums
and never overwritten; a change of rules or config needs a new graph version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import osmnx
import yaml

from graph_llm.graph.carriageways import classify_all_streets
from graph_llm.graph.junctions import build_junctions, place_street_ids
from graph_llm.graph.processed import keep_largest_strongly_connected_component
from graph_llm.graph.restrictions import load_restrictions
from graph_llm.graph.simplify import simplify_keeping_roads_apart
from graph_llm.graph.snapshot import compute_sha256, load_snapshot
from graph_llm.graph.streets import StreetLayer, build_street_layer
from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.junction import Junction
from graph_llm.schemas.reference_graph import ReferenceGraphConfig, ReferenceGraphMetadata
from graph_llm.schemas.restriction import TurnRestriction
from graph_llm.schemas.review import ReviewDecisionKind, ReviewLog
from graph_llm.schemas.street import Street, StreetCatalog

GRAPH_FILE_NAME = "graph.graphml"
STREETS_FILE_NAME = "streets.json"
JUNCTIONS_FILE_NAME = "junctions.json"
METADATA_FILE_NAME = "metadata.json"


@dataclass(frozen=True)
class ReferenceGraph:
    """A reference graph, built or loaded from disk."""

    graph: RoadGraph
    streets: list[Street]
    junctions: list[Junction]
    restrictions: list[TurnRestriction]
    metadata: ReferenceGraphMetadata


def load_reference_config(config_path: Path) -> ReferenceGraphConfig:
    """Read and validate a reference graph config from a YAML file."""
    return ReferenceGraphConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )


def build_reference_graph(
    config: ReferenceGraphConfig, raw_data_directory: Path
) -> tuple[RoadGraph, StreetLayer, list[Junction], list[TurnRestriction], ReferenceGraphMetadata]:
    """Run every processing step; nothing is written."""
    area_id = config.streets.source_area_id
    snapshot_graph, snapshot_metadata = load_snapshot(raw_data_directory, area_id)
    restrictions, restriction_metadata = load_restrictions(raw_data_directory, area_id)
    simplified = simplify_keeping_roads_apart(snapshot_graph, restrictions)
    graph, removed_node_ids = keep_largest_strongly_connected_component(simplified)
    street_layer = build_street_layer(graph, config.streets)
    labels = classify_all_streets(
        graph, street_layer.streets, config.junctions.carriageway_max_separation_m
    )
    junctions = build_junctions(graph, street_layer.streets, labels, config.junctions)
    metadata = ReferenceGraphMetadata(
        config=config,
        source_snapshot=snapshot_metadata,
        source_restrictions=restriction_metadata,
        created_at=datetime.now(UTC),
        osmnx_version=osmnx.__version__,
        snapshot_node_count=snapshot_graph.number_of_nodes(),
        snapshot_edge_count=snapshot_graph.number_of_edges(),
        simplified_node_count=simplified.number_of_nodes(),
        simplified_edge_count=simplified.number_of_edges(),
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        removed_node_ids=removed_node_ids,
        removed_edge_count=simplified.number_of_edges() - graph.number_of_edges(),
        unnamed_edge_count=street_layer.unnamed_edge_count,
        street_count=len(street_layer.streets),
        place_count=len(place_street_ids(street_layer.streets, config.junctions)),
        junction_count=len(junctions),
        name_resolutions=street_layer.name_resolutions,
        graphml_sha256="",
        streets_sha256="",
        junctions_sha256="",
    )
    return graph, street_layer, junctions, restrictions, metadata


def reference_graph_directory(processed_data_directory: Path, graph_version: str) -> Path:
    """Return the directory of one reference graph version."""
    return processed_data_directory / graph_version


def save_reference_graph(
    graph: RoadGraph,
    streets: list[Street],
    junctions: list[Junction],
    metadata: ReferenceGraphMetadata,
    processed_data_directory: Path,
) -> ReferenceGraphMetadata:
    """Write a reference graph with checksums.

    Raises:
        FileExistsError: If this graph version already exists.
    """
    graph_version = metadata.config.streets.graph_version
    target = reference_graph_directory(processed_data_directory, graph_version)
    if target.exists():
        raise FileExistsError(f"{target} exists; reference graphs are immutable.")
    target.mkdir(parents=True)
    osmnx.save_graphml(graph, target / GRAPH_FILE_NAME)
    catalog = StreetCatalog(graph_version=graph_version, streets=streets)
    (target / STREETS_FILE_NAME).write_text(catalog.model_dump_json(indent=2), encoding="utf-8")
    (target / JUNCTIONS_FILE_NAME).write_text(
        json.dumps([j.model_dump() for j in junctions], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    final = metadata.model_copy(
        update={
            "graphml_sha256": compute_sha256(target / GRAPH_FILE_NAME),
            "streets_sha256": compute_sha256(target / STREETS_FILE_NAME),
            "junctions_sha256": compute_sha256(target / JUNCTIONS_FILE_NAME),
        }
    )
    (target / METADATA_FILE_NAME).write_text(final.model_dump_json(indent=2), encoding="utf-8")
    return final


def load_reference_graph(
    processed_data_directory: Path, raw_data_directory: Path, graph_version: str
) -> ReferenceGraph:
    """Load a frozen reference graph and its restrictions, verifying every checksum."""
    source = reference_graph_directory(processed_data_directory, graph_version)
    metadata = ReferenceGraphMetadata.model_validate_json(
        (source / METADATA_FILE_NAME).read_text(encoding="utf-8")
    )
    for name, expected in (
        (GRAPH_FILE_NAME, metadata.graphml_sha256),
        (STREETS_FILE_NAME, metadata.streets_sha256),
        (JUNCTIONS_FILE_NAME, metadata.junctions_sha256),
    ):
        if compute_sha256(source / name) != expected:
            raise ValueError(f"Checksum mismatch for {source / name}: the graph was modified.")
    catalog = StreetCatalog.model_validate_json(
        (source / STREETS_FILE_NAME).read_text(encoding="utf-8")
    )
    junctions = [
        Junction.model_validate(item)
        for item in json.loads((source / JUNCTIONS_FILE_NAME).read_text(encoding="utf-8"))
    ]
    restrictions, _ = load_restrictions(raw_data_directory, metadata.config.streets.source_area_id)
    return ReferenceGraph(
        graph=osmnx.load_graphml(source / GRAPH_FILE_NAME),
        streets=catalog.streets,
        junctions=junctions,
        restrictions=restrictions,
        metadata=metadata,
    )


def load_review_log(path: Path) -> ReviewLog:
    """Read a review log from YAML."""
    return ReviewLog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def excluded_item_ids(review_log: ReviewLog | None) -> frozenset[str]:
    """Return the street and junction ids that the review excluded from questions."""
    if review_log is None:
        return frozenset()
    return frozenset(
        decision.item_id
        for decision in review_log.decisions
        if decision.decision is ReviewDecisionKind.EXCLUDED_FROM_QUESTIONS
    )
