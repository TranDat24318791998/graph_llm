"""Hand-built cases for the graph v2 rules: carriageways and junctions.

Coordinates are in degrees near the study area; 0.001 degree is about 111 m.
"""

import networkx as nx
import pytest

from conftest import CONFIG
from graph_llm.graph.carriageways import (
    EdgeDirection,
    StreetDirection,
    centre_line_length_m,
    classify_all_streets,
    street_direction,
)
from graph_llm.graph.junctions import build_junctions, place_street_ids
from graph_llm.graph.streets import build_street_layer
from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.junction import JunctionSettings
from graph_llm.schemas.question import EdgeRef
from graph_llm.schemas.street import Street

SETTINGS = JunctionSettings()
STEP = 0.001
MEDIAN = 0.00018  # about 20 m


def add_node(graph: RoadGraph, node: int, x: float, y: float) -> None:
    graph.add_node(node, x=106.7 + x, y=10.77 + y)


def add_edge(
    graph: RoadGraph, source: int, target: int, name: str | None, length: float, **tags: object
) -> None:
    attributes: dict[str, object] = {"length": length, "oneway": True, "highway": "primary"}
    if name is not None:
        attributes["name"] = name
    attributes.update(tags)
    graph.add_edge(source, target, **attributes)


def add_two_way(
    graph: RoadGraph, first: int, second: int, name: str | None, length: float, **tags: object
) -> None:
    add_edge(graph, first, second, name, length, oneway=False, **tags)
    add_edge(graph, second, first, name, length, oneway=False, **tags)


def streets_and_labels(
    graph: RoadGraph,
) -> tuple[dict[str, Street], dict[str, dict[EdgeRef, EdgeDirection]]]:
    streets = build_street_layer(graph, CONFIG).streets
    labels = classify_all_streets(graph, streets)
    return {street.display_name: street for street in streets}, labels


def divided_road_with_crossing() -> RoadGraph:
    """Divided road "Lê Lợi": eastbound carriageway 1 -> 2 -> 3 and westbound 13 -> 12 -> 11,
    20 m apart. Two-way "Pasteur" crosses both carriageways: 20 <-> 2 <-> 12 <-> 30."""
    graph: RoadGraph = nx.MultiDiGraph()
    for node, x, y in ((1, 0.0, 0.0), (2, STEP, 0.0), (3, 2 * STEP, 0.0)):
        add_node(graph, node, x, y)
    for node, x, y in ((11, 0.0, -MEDIAN), (12, STEP, -MEDIAN), (13, 2 * STEP, -MEDIAN)):
        add_node(graph, node, x, y)
    add_node(graph, 20, STEP, STEP)
    add_node(graph, 30, STEP, -STEP)
    add_edge(graph, 1, 2, "Lê Lợi", 111.0)
    add_edge(graph, 2, 3, "Lê Lợi", 111.0)
    add_edge(graph, 13, 12, "Lê Lợi", 111.0)
    add_edge(graph, 12, 11, "Lê Lợi", 111.0)
    add_two_way(graph, 20, 2, "Pasteur", 111.0)
    add_two_way(graph, 2, 12, "Pasteur", 20.0)
    add_two_way(graph, 12, 30, "Pasteur", 91.0)
    return graph


def test_carriageways_of_a_divided_road_are_paired() -> None:
    graph = divided_road_with_crossing()
    streets, labels = streets_and_labels(graph)

    le_loi = streets["Lê Lợi"]
    le_loi_labels = labels[le_loi.street_id]
    assert set(le_loi_labels.values()) == {EdgeDirection.DIVIDED}
    assert street_direction(le_loi_labels) is StreetDirection.TWO_WAY_DIVIDED
    lengths = {edge: 111.0 for edge in le_loi.edges}
    assert centre_line_length_m(le_loi_labels, lengths) == pytest.approx(222.0)
    pasteur_labels = labels[streets["Pasteur"].street_id]
    assert street_direction(pasteur_labels) is StreetDirection.TWO_WAY


def test_median_crossing_makes_one_junction() -> None:
    graph = divided_road_with_crossing()
    streets, labels = streets_and_labels(graph)

    junctions = build_junctions(graph, list(streets.values()), labels, SETTINGS)

    crossing = [j for j in junctions if 2 in j.node_ids]
    assert len(crossing) == 1
    assert set(crossing[0].node_ids) == {2, 12}
    assert {streets["Lê Lợi"].street_id, streets["Pasteur"].street_id} <= set(
        crossing[0].street_ids
    )


def test_one_way_street_and_a_flip_in_direction() -> None:
    graph: RoadGraph = nx.MultiDiGraph()
    for node in range(1, 5):
        add_node(graph, node, node * STEP, 0)
    add_edge(graph, 1, 2, "Pasteur", 111.0)
    add_edge(graph, 2, 3, "Pasteur", 111.0)
    add_edge(graph, 4, 3, "Pasteur", 111.0)
    streets, labels = streets_and_labels(graph)
    pasteur = streets["Pasteur"]
    first_two = {
        e: labels[pasteur.street_id][e]
        for e in pasteur.edges
        if e.target_node_id != 3 or e.source_node_id == 2
    }

    assert street_direction(first_two) is StreetDirection.ONE_WAY
    assert street_direction(labels[pasteur.street_id]) is StreetDirection.MIXED


def test_streets_meet_at_a_roundabout_through_unnamed_ring_edges() -> None:
    """Ring 1 -> 2 -> 3 -> 4 -> 1 (unnamed roundabout); "Lê Duẩn" reaches node 1 and
    "Phạm Ngọc Thạch" reaches node 3, so they share no node but meet at the roundabout."""
    graph: RoadGraph = nx.MultiDiGraph()
    ring = {1: (0, 0.0002), 2: (0.0002, 0), 3: (0, -0.0002), 4: (-0.0002, 0)}
    for node, (x, y) in ring.items():
        add_node(graph, node, x, y)
    for source, target in ((1, 2), (2, 3), (3, 4), (4, 1)):
        add_edge(graph, source, target, None, 30.0, junction="roundabout")
    add_node(graph, 10, 0, STEP)
    add_node(graph, 30, 0, -STEP)
    add_two_way(graph, 10, 1, "Lê Duẩn", 90.0)
    add_two_way(graph, 3, 30, "Phạm Ngọc Thạch", 90.0)
    streets, labels = streets_and_labels(graph)

    junctions = build_junctions(graph, list(streets.values()), labels, SETTINGS)

    ring_junction = [j for j in junctions if 1 in j.node_ids][0]
    assert set(ring_junction.node_ids) == {1, 2, 3, 4}
    assert {streets["Lê Duẩn"].street_id, streets["Phạm Ngọc Thạch"].street_id} == set(
        ring_junction.street_ids
    )


def test_bridge_over_a_street_is_not_a_junction() -> None:
    """ "Cầu A" passes over "Đường B" with no shared node; their nodes are 5 m apart."""
    graph: RoadGraph = nx.MultiDiGraph()
    add_node(graph, 1, -STEP, 0)
    add_node(graph, 2, 0, 0)
    add_node(graph, 3, STEP, 0)
    add_node(graph, 4, 0.00004, -STEP)
    add_node(graph, 5, 0.00004, 0.00004)
    add_node(graph, 6, 0.00004, STEP)
    add_node(graph, 7, -STEP, STEP)
    add_two_way(graph, 1, 2, "Cầu A", 111.0, bridge="yes")
    add_two_way(graph, 2, 3, "Cầu A", 111.0, bridge="yes")
    add_two_way(graph, 4, 5, "Đường B", 111.0)
    add_two_way(graph, 5, 6, "Đường B", 111.0)
    add_two_way(graph, 5, 7, "Đường C", 150.0)
    streets, labels = streets_and_labels(graph)

    junctions = build_junctions(graph, list(streets.values()), labels, SETTINGS)

    bridge = streets["Cầu A"].street_id
    assert all(bridge not in j.street_ids for j in junctions if len(j.street_ids) > 1)


def test_unnamed_slip_lane_joins_its_junction() -> None:
    """A 30 m unnamed primary_link 2 -> 5 connects "Hàm Nghi" and "Pasteur" ends."""
    graph: RoadGraph = nx.MultiDiGraph()
    add_node(graph, 1, -STEP, 0)
    add_node(graph, 2, 0, 0)
    add_node(graph, 3, STEP, 0)
    add_node(graph, 5, 0.0002, -0.0002)
    add_node(graph, 6, 0.0002, -STEP)
    add_node(graph, 7, 0.0004, -STEP)
    add_two_way(graph, 1, 2, "Hàm Nghi", 111.0)
    add_two_way(graph, 2, 3, "Hàm Nghi", 111.0)
    add_two_way(graph, 5, 6, "Pasteur", 90.0)
    add_two_way(graph, 5, 7, "Nam Kỳ Khởi Nghĩa", 90.0)
    add_edge(graph, 2, 5, None, 30.0, highway="primary_link")
    add_edge(graph, 5, 2, None, 30.0, highway="primary_link")
    streets, labels = streets_and_labels(graph)

    junctions = build_junctions(graph, list(streets.values()), labels, SETTINGS)

    joined = [j for j in junctions if 2 in j.node_ids][0]
    assert set(joined.node_ids) == {2, 5}
    assert streets["Pasteur"].street_id in joined.street_ids
    assert streets["Hàm Nghi"].street_id in joined.street_ids


def test_hairpin_is_not_a_divided_road() -> None:
    """A one-way street that turns back on itself (2 -> 3 then 3 -> 4, 2 m alongside)
    shares node 3 between the two opposite edges: a hairpin, not two carriageways."""
    graph: RoadGraph = nx.MultiDiGraph()
    add_node(graph, 1, 0.0, 0.0)
    add_node(graph, 2, STEP, 0.0)
    add_node(graph, 3, 2 * STEP, 0.0)
    add_node(graph, 4, STEP, 0.00002)
    add_edge(graph, 1, 2, "Phó Đức Chính", 111.0)
    add_edge(graph, 2, 3, "Phó Đức Chính", 111.0)
    add_edge(graph, 3, 4, "Phó Đức Chính", 111.0)
    streets, labels = streets_and_labels(graph)

    assert EdgeDirection.DIVIDED not in set(labels[streets["Phó Đức Chính"].street_id].values())


def test_a_place_drawn_in_two_pieces_is_one_junction() -> None:
    """ "Công trường X" has two separate named pieces (1-2 and 3-4); the streets around
    them meet at one junction."""
    graph: RoadGraph = nx.MultiDiGraph()
    for node, x, y in ((1, 0.0, 0.0), (2, 0.0005, 0.0), (3, 0.0, 0.0003), (4, 0.0005, 0.0003)):
        add_node(graph, node, x, y)
    add_node(graph, 10, -STEP, 0.0)
    add_node(graph, 11, -STEP, 0.0003)
    add_edge(graph, 1, 2, "Công trường X", 60.0)
    add_edge(graph, 4, 3, "Công trường X", 60.0)
    add_two_way(graph, 2, 4, "Đường Đông", 33.0)
    add_two_way(graph, 3, 1, "Đường Tây", 33.0)
    add_two_way(graph, 10, 1, "Đường Nam", 111.0)
    add_two_way(graph, 11, 3, "Đường Bắc", 111.0)
    streets, labels = streets_and_labels(graph)

    junctions = build_junctions(graph, list(streets.values()), labels, SETTINGS)

    place = [j for j in junctions if j.place_names == ["Công trường X"]]
    assert len(place) == 1
    assert {1, 2, 3, 4} <= set(place[0].node_ids)


def test_corners_of_one_crossing_are_merged() -> None:
    """ "Hàm Nghi" meets "Pasteur" at node 2 and, through a 30 m slip lane of Pasteur
    (named), again at node 5: both node groups have the same two streets and are
    joined by a short edge, so they are one crossing."""
    graph: RoadGraph = nx.MultiDiGraph()
    add_node(graph, 1, -STEP, 0.0)
    add_node(graph, 2, 0.0, 0.0)
    add_node(graph, 5, 0.0002, 0.0)
    add_node(graph, 3, STEP, 0.0)
    add_node(graph, 6, 0.0, -STEP)
    add_node(graph, 7, 0.0002, -STEP)
    add_two_way(graph, 1, 2, "Hàm Nghi", 111.0)
    add_two_way(graph, 2, 5, "Hàm Nghi", 22.0)
    add_two_way(graph, 5, 3, "Hàm Nghi", 89.0)
    add_two_way(graph, 2, 6, "Pasteur", 111.0)
    add_two_way(graph, 5, 7, "Pasteur", 111.0)
    streets, labels = streets_and_labels(graph)

    junctions = build_junctions(graph, list(streets.values()), labels, SETTINGS)

    crossing = [j for j in junctions if 2 in j.node_ids][0]
    assert {2, 5} <= set(crossing.node_ids)


def test_street_like_place_is_a_street() -> None:
    graph: RoadGraph = nx.MultiDiGraph()
    add_node(graph, 1, 0.0, 0.0)
    add_node(graph, 2, 0.002, 0.0)
    add_two_way(graph, 1, 2, "Công trường Lam Sơn", 200.0)
    streets, _ = streets_and_labels(graph)
    settings = SETTINGS.model_copy(update={"street_like_places": ["Công trường Lam Sơn"]})

    assert place_street_ids(list(streets.values()), SETTINGS)
    assert not place_street_ids(list(streets.values()), settings)
