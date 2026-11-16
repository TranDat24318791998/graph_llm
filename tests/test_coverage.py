import networkx as nx
import pytest

from graph_llm.graph.coverage import compute_coverage_report, has_value
from graph_llm.graph.types import RoadGraph


def build_hand_checked_graph() -> RoadGraph:
    """Four edges with a total length of 1000 m.

    1 -> 2  Le Loi, 100 m, one-way, primary
    2 -> 3  Pasteur, 200 m, two-way, secondary, 2 lanes
    3 -> 2  Pasteur, 200 m, two-way, secondary, 2 lanes
    3 -> 4  unnamed, 500 m, two-way, merged classes residential and service
    """
    graph: RoadGraph = nx.MultiDiGraph()
    graph.add_edge(1, 2, name="Le Loi", length=100.0, oneway=True, highway="primary")
    graph.add_edge(2, 3, name="Pasteur", length=200.0, oneway=False, highway="secondary", lanes="2")
    graph.add_edge(3, 2, name="Pasteur", length=200.0, oneway=False, highway="secondary", lanes="2")
    graph.add_edge(3, 4, length=500.0, oneway=False, highway=["residential", "service"])
    return graph


def test_graph_statistics() -> None:
    report = compute_coverage_report(build_hand_checked_graph(), "test_area")

    assert report.node_count == 4
    assert report.edge_count == 4
    assert report.total_length_m == pytest.approx(1000.0)
    assert report.oneway_edge_count == 1
    assert report.distinct_street_name_count == 2


def test_attribute_coverage_by_edges_and_by_length() -> None:
    report = compute_coverage_report(build_hand_checked_graph(), "test_area")
    coverage_by_attribute = {coverage.attribute: coverage for coverage in report.attributes}

    name_coverage = coverage_by_attribute["name"]
    assert name_coverage.edges_with_value == 3
    assert name_coverage.edge_share == pytest.approx(0.75)
    assert name_coverage.length_share == pytest.approx(0.5)

    lanes_coverage = coverage_by_attribute["lanes"]
    assert lanes_coverage.edges_with_value == 2
    assert lanes_coverage.length_share == pytest.approx(0.4)

    assert coverage_by_attribute["maxspeed"].edges_with_value == 0


def test_multi_valued_attribute_is_counted_per_class() -> None:
    report = compute_coverage_report(build_hand_checked_graph(), "test_area")
    coverage_by_attribute = {coverage.attribute: coverage for coverage in report.attributes}

    assert coverage_by_attribute["highway"].multi_valued_edges == 1
    assert report.highway_class_edge_counts == {
        "secondary": 2,
        "primary": 1,
        "residential": 1,
        "service": 1,
    }


@pytest.mark.parametrize("empty_value", [None, "", [], float("nan")])
def test_empty_values_do_not_count_as_coverage(empty_value: object) -> None:
    assert has_value(empty_value) is False


@pytest.mark.parametrize("real_value", ["Pasteur", False, 0, ["a", "b"]])
def test_real_values_count_as_coverage(real_value: object) -> None:
    assert has_value(real_value) is True
