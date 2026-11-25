"""Shared test fixtures: a small road graph whose answers are checked by hand."""

import networkx as nx
import pytest

from graph_llm.graph.streets import build_street_layer
from graph_llm.graph.types import RoadGraph
from graph_llm.questions.graph_view import GraphView
from graph_llm.schemas.street import GraphProcessingConfig

CONFIG = GraphProcessingConfig(
    graph_version="test_p1",
    source_area_id="test_area",
    merge_gap_m=50.0,
    internal_road_names=[],
    street_overrides=[],
    name_aliases=[],
)


def position(column: int, row: int) -> dict[str, float]:
    return {"x": 106.7 + column * 0.001, "y": 10.7 + row * 0.001}


def build_hand_checked_view() -> GraphView:
    """A small grid, every edge 100 m.

        8           4           6
        |           |           |
        1 --------> 2 --------> 3        Pasteur: one-way 1 -> 2 -> 3
                    |           |
                    5           7
                    |
                    9 <-------- 10       Đường 10A: one-way 10 -> 9

    Nguyễn Huệ: 1-8 two-way.  Lê Lợi: 4-2-5-9 two-way, 2 lanes.
    Hai Bà Trưng: 6-3-7-10 two-way.
    """
    graph: RoadGraph = nx.MultiDiGraph()
    coordinates = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (1, 1), 5: (1, -1), 6: (2, 1)}
    coordinates |= {7: (2, -1), 8: (0, 1), 9: (1, -2), 10: (2, -2)}
    for node, (column, row) in coordinates.items():
        graph.add_node(node, **position(column, row))

    def one_way(source: int, target: int, name: str) -> None:
        graph.add_edge(source, target, length=100.0, oneway=True, name=name)

    def two_way(first: int, second: int, name: str, lanes: str | None = None) -> None:
        for source, target in ((first, second), (second, first)):
            attributes: dict[str, object] = {"length": 100.0, "oneway": False, "name": name}
            if lanes is not None:
                attributes["lanes"] = lanes
            graph.add_edge(source, target, **attributes)

    one_way(1, 2, "Pasteur")
    one_way(2, 3, "Pasteur")
    one_way(10, 9, "Đường 10A")
    two_way(1, 8, "Nguyễn Huệ")
    for first, second in ((4, 2), (2, 5), (5, 9)):
        two_way(first, second, "Lê Lợi", lanes="2")
    for first, second in ((6, 3), (3, 7), (7, 10)):
        two_way(first, second, "Hai Bà Trưng")
    layer = build_street_layer(graph, CONFIG)
    return GraphView.from_graph_and_streets(graph, layer.streets)


@pytest.fixture
def view() -> GraphView:
    return build_hand_checked_view()
