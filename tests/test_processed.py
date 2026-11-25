from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import pytest

from graph_llm.graph.processed import (
    keep_largest_strongly_connected_component,
    load_processed_graph,
    load_processing_config,
    process_snapshot,
    save_processed_graph,
)
from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.graph import BoundingBox, SnapshotMetadata, StudyArea
from graph_llm.schemas.street import GraphProcessingConfig

PHASE_ONE_CONFIG_PATH = Path("configs/graphs/hcmc_district1_v2_p1.yaml")


def build_graph_with_dead_end() -> RoadGraph:
    """A two-way triangle 1-2-3 and a one-way exit 3 -> 4 with no way back.

    Node 4 can be reached but not left, so it is outside the largest strongly
    connected component.
    """
    graph: RoadGraph = nx.MultiDiGraph(crs="epsg:4326")
    for node in range(1, 5):
        graph.add_node(node, x=106.7 + node * 0.001, y=10.7)
    for source, target in ((1, 2), (2, 3), (3, 1)):
        graph.add_edge(source, target, length=100.0, name="Pasteur", oneway=False)
        graph.add_edge(target, source, length=100.0, name="Pasteur", oneway=False)
    graph.add_edge(3, 4, length=50.0, name="Lê Lợi", oneway=True)
    return graph


def make_config() -> GraphProcessingConfig:
    return GraphProcessingConfig(
        graph_version="test_p1",
        source_area_id="test_area",
        merge_gap_m=200.0,
        internal_road_names=[],
        street_overrides=[],
        name_aliases=[],
    )


def make_snapshot_metadata() -> SnapshotMetadata:
    area = StudyArea(
        area_id="test_area",
        description="Test",
        bounding_box=BoundingBox(west=106.7, south=10.69, east=106.71, north=10.71),
        network_type="drive",
        simplify=True,
    )
    return SnapshotMetadata(
        area=area,
        downloaded_at=datetime(2026, 11, 24, tzinfo=UTC),
        osmnx_version="test",
        node_count=4,
        edge_count=7,
        graphml_sha256="0" * 64,
    )


def test_phase_one_config_is_valid() -> None:
    config = load_processing_config(PHASE_ONE_CONFIG_PATH)

    assert config.source_area_id == "hcmc_district1_v2"
    assert config.merge_gap_m == 200.0


def test_dead_end_is_removed() -> None:
    graph, removed_node_ids = keep_largest_strongly_connected_component(build_graph_with_dead_end())

    assert removed_node_ids == [4]
    assert sorted(graph.nodes) == [1, 2, 3]
    assert graph.number_of_edges() == 6


def test_street_on_removed_edges_disappears() -> None:
    result = process_snapshot(build_graph_with_dead_end(), make_config())

    assert [street.street_id for street in result.street_layer.streets] == ["pasteur"]
    assert result.removed_edge_count == 1


def test_processed_graph_round_trips_and_is_frozen(tmp_path: Path) -> None:
    config = make_config()
    result = process_snapshot(build_graph_with_dead_end(), config)
    metadata = save_processed_graph(result, config, make_snapshot_metadata(), tmp_path)

    loaded = load_processed_graph(tmp_path, "test_p1")
    assert loaded.metadata == metadata
    assert loaded.streets == result.street_layer.streets
    assert sorted(loaded.graph.nodes) == [1, 2, 3]
    assert loaded.graph.number_of_edges() == 6

    with pytest.raises(FileExistsError):
        save_processed_graph(result, config, make_snapshot_metadata(), tmp_path)


def test_modified_streets_file_is_detected(tmp_path: Path) -> None:
    config = make_config()
    result = process_snapshot(build_graph_with_dead_end(), config)
    save_processed_graph(result, config, make_snapshot_metadata(), tmp_path)
    streets_path = tmp_path / "test_p1" / "streets.json"
    streets_path.write_text(streets_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_processed_graph(tmp_path, "test_p1")
