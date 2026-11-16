"""Download the OSM road graph once and freeze it on disk.

OpenStreetMap changes every day. To keep every later result reproducible, the
graph is downloaded a single time, stored with its provenance, and never
downloaded again for the same area id.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import osmnx
import yaml

from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.graph import SnapshotMetadata, StudyArea

GRAPH_FILE_NAME = "graph.graphml"
METADATA_FILE_NAME = "metadata.json"


def load_study_area(config_path: Path) -> StudyArea:
    """Read and validate a study area definition from a YAML file."""
    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    return StudyArea.model_validate(raw_config)


def snapshot_directory(raw_data_directory: Path, area_id: str) -> Path:
    """Return the directory that holds the snapshot of one area."""
    return raw_data_directory / area_id


def compute_sha256(file_path: Path) -> str:
    """Return the SHA-256 checksum of a file as a hex string."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_snapshot(area: StudyArea, raw_data_directory: Path) -> SnapshotMetadata:
    """Download the road graph of an area and store it with its metadata.

    Raises:
        FileExistsError: If a snapshot for this area id already exists. A frozen
            snapshot is never overwritten; use a new area id for a new download.
    """
    target_directory = snapshot_directory(raw_data_directory, area.area_id)
    if target_directory.exists():
        raise FileExistsError(
            f"A snapshot already exists at {target_directory}. "
            "Snapshots are immutable; choose a new area_id to download again."
        )

    box = area.bounding_box
    graph = osmnx.graph_from_bbox(
        bbox=(box.west, box.south, box.east, box.north),
        network_type=area.network_type,
        simplify=area.simplify,
    )

    target_directory.mkdir(parents=True)
    graph_path = target_directory / GRAPH_FILE_NAME
    osmnx.save_graphml(graph, graph_path)

    metadata = SnapshotMetadata(
        area=area,
        downloaded_at=datetime.now(UTC),
        osmnx_version=osmnx.__version__,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        graphml_sha256=compute_sha256(graph_path),
    )
    metadata_path = target_directory / METADATA_FILE_NAME
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return metadata


def load_snapshot(raw_data_directory: Path, area_id: str) -> tuple[RoadGraph, SnapshotMetadata]:
    """Load a frozen snapshot and verify that the graph file is unchanged.

    Raises:
        ValueError: If the stored graph file does not match its recorded checksum.
    """
    source_directory = snapshot_directory(raw_data_directory, area_id)
    metadata_text = (source_directory / METADATA_FILE_NAME).read_text(encoding="utf-8")
    metadata = SnapshotMetadata.model_validate_json(metadata_text)

    graph_path = source_directory / GRAPH_FILE_NAME
    actual_checksum = compute_sha256(graph_path)
    if actual_checksum != metadata.graphml_sha256:
        raise ValueError(f"Checksum mismatch for {graph_path}: the frozen snapshot was modified.")

    graph = osmnx.load_graphml(graph_path)
    return graph, metadata
