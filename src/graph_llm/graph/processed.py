"""Turn a frozen snapshot into the processed graph and store it, frozen as well.

The processed graph is the road graph reduced to its largest strongly connected
component, so that every node can reach every other node along legal edge
directions, together with its street layer. A processed graph version is written
once and never overwritten; a change of the processing config needs a new version.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import osmnx
import yaml

from graph_llm.graph.snapshot import compute_sha256
from graph_llm.graph.streets import StreetLayer, build_street_layer
from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.graph import SnapshotMetadata
from graph_llm.schemas.street import (
    GraphProcessingConfig,
    ProcessedGraphMetadata,
    Street,
    StreetCatalog,
)

GRAPH_FILE_NAME = "graph.graphml"
STREETS_FILE_NAME = "streets.json"
METADATA_FILE_NAME = "metadata.json"


@dataclass(frozen=True)
class ProcessingResult:
    """The output of processing, before it is written to disk."""

    graph: RoadGraph
    street_layer: StreetLayer
    removed_node_ids: list[int]
    removed_edge_count: int


@dataclass(frozen=True)
class ProcessedGraph:
    """A processed graph loaded from disk."""

    graph: RoadGraph
    streets: list[Street]
    metadata: ProcessedGraphMetadata


def load_processing_config(config_path: Path) -> GraphProcessingConfig:
    """Read and validate a graph processing config from a YAML file."""
    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    return GraphProcessingConfig.model_validate(raw_config)


def keep_largest_strongly_connected_component(graph: RoadGraph) -> tuple[RoadGraph, list[int]]:
    """Return the subgraph on the largest strongly connected component and the removed nodes."""
    largest_component = max(nx.strongly_connected_components(graph), key=len)
    kept_graph: RoadGraph = graph.subgraph(largest_component).copy()
    removed_node_ids = sorted(node for node in graph.nodes if node not in largest_component)
    return kept_graph, removed_node_ids


def process_snapshot(snapshot_graph: RoadGraph, config: GraphProcessingConfig) -> ProcessingResult:
    """Reduce the snapshot to its largest strongly connected component and build its streets."""
    graph, removed_node_ids = keep_largest_strongly_connected_component(snapshot_graph)
    return ProcessingResult(
        graph=graph,
        street_layer=build_street_layer(graph, config),
        removed_node_ids=removed_node_ids,
        removed_edge_count=snapshot_graph.number_of_edges() - graph.number_of_edges(),
    )


def processed_graph_directory(processed_data_directory: Path, graph_version: str) -> Path:
    """Return the directory that holds one processed graph version."""
    return processed_data_directory / graph_version


def save_processed_graph(
    result: ProcessingResult,
    config: GraphProcessingConfig,
    source_snapshot: SnapshotMetadata,
    processed_data_directory: Path,
) -> ProcessedGraphMetadata:
    """Write the graph, its streets and its metadata.

    Raises:
        FileExistsError: If this graph version already exists. A processed graph is
            never overwritten; use a new graph version instead.
    """
    target_directory = processed_graph_directory(processed_data_directory, config.graph_version)
    if target_directory.exists():
        raise FileExistsError(
            f"A processed graph already exists at {target_directory}. "
            "Processed graphs are immutable; choose a new graph_version."
        )
    target_directory.mkdir(parents=True)

    graph_path = target_directory / GRAPH_FILE_NAME
    osmnx.save_graphml(result.graph, graph_path)

    catalog = StreetCatalog(graph_version=config.graph_version, streets=result.street_layer.streets)
    streets_path = target_directory / STREETS_FILE_NAME
    streets_path.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")

    metadata = ProcessedGraphMetadata(
        config=config,
        source_snapshot=source_snapshot,
        created_at=datetime.now(UTC),
        osmnx_version=osmnx.__version__,
        node_count=result.graph.number_of_nodes(),
        edge_count=result.graph.number_of_edges(),
        removed_node_ids=result.removed_node_ids,
        removed_edge_count=result.removed_edge_count,
        unnamed_edge_count=result.street_layer.unnamed_edge_count,
        street_count=len(result.street_layer.streets),
        name_resolutions=result.street_layer.name_resolutions,
        graphml_sha256=compute_sha256(graph_path),
        streets_sha256=compute_sha256(streets_path),
    )
    metadata_path = target_directory / METADATA_FILE_NAME
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return metadata


def load_processed_graph(processed_data_directory: Path, graph_version: str) -> ProcessedGraph:
    """Load a processed graph and verify that its files are unchanged.

    Raises:
        ValueError: If a stored file does not match its recorded checksum.
    """
    source_directory = processed_graph_directory(processed_data_directory, graph_version)
    metadata_text = (source_directory / METADATA_FILE_NAME).read_text(encoding="utf-8")
    metadata = ProcessedGraphMetadata.model_validate_json(metadata_text)

    graph_path = source_directory / GRAPH_FILE_NAME
    streets_path = source_directory / STREETS_FILE_NAME
    for path, expected_checksum in (
        (graph_path, metadata.graphml_sha256),
        (streets_path, metadata.streets_sha256),
    ):
        if compute_sha256(path) != expected_checksum:
            raise ValueError(f"Checksum mismatch for {path}: the processed graph was modified.")

    catalog = StreetCatalog.model_validate_json(streets_path.read_text(encoding="utf-8"))
    graph = osmnx.load_graphml(graph_path)
    return ProcessedGraph(graph=graph, streets=catalog.streets, metadata=metadata)
