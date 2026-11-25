import random

import networkx as nx
import pytest

from conftest import CONFIG, position
from graph_llm.graph.carriageways import StreetDirection
from graph_llm.graph.streets import build_street_layer
from graph_llm.graph.types import RoadGraph
from graph_llm.questions.generate import (
    l1_length_candidates,
    l3_next_on_oneway_candidates,
    l5_common_neighbours_candidates,
    sample_candidates,
    slot_name,
    walk_to_next_street,
)
from graph_llm.questions.graph_view import GraphView
from graph_llm.questions.routes import (
    route_street_sequence,
    shortest_length_following_sequence,
    shortest_length_ignoring_direction,
    shortest_route,
)
from graph_llm.schemas.question import StreetNameAnswer


def test_direction(view: GraphView) -> None:

    assert view.direction("pasteur") is StreetDirection.ONE_WAY
    assert view.direction("le-loi") is StreetDirection.TWO_WAY


def test_opposite_one_way_pieces_far_apart_are_mixed() -> None:
    """Two opposite one-way pieces 111 m apart are not two carriageways of one road."""
    graph: RoadGraph = nx.MultiDiGraph()
    for node, (column, row) in {1: (0, 0), 2: (1, 0), 3: (0, 1), 4: (1, 1)}.items():
        graph.add_node(node, **position(column, row))
    graph.add_edge(1, 2, length=100.0, oneway=True, name="Lê Duẩn")
    graph.add_edge(4, 3, length=100.0, oneway=True, name="Lê Duẩn")
    graph.add_edge(2, 4, length=30.0, oneway=False, highway="unnamed")
    graph.add_edge(4, 2, length=30.0, oneway=False, highway="unnamed")
    view = GraphView.from_graph_and_streets(graph, build_street_layer(graph, CONFIG).streets)

    assert view.direction("le-duan") is StreetDirection.MIXED


def test_lane_count(view: GraphView) -> None:

    assert view.lane_count("le-loi") == 2
    assert view.lane_count("pasteur") is None


def test_regular_neighbours(view: GraphView) -> None:

    assert view.regular_neighbours("pasteur") == {
        "nguyen-hue": {1},
        "le-loi": {2},
        "hai-ba-trung": {3},
    }


def test_walk_follows_the_one_way_direction(view: GraphView) -> None:

    forward = walk_to_next_street(view, "pasteur", {2}, {"le-loi"}, forward=True)
    backward = walk_to_next_street(view, "pasteur", {2}, {"le-loi"}, forward=False)

    assert forward is not None and forward[0] == "hai-ba-trung" and forward[2] == {3}
    assert backward is not None and backward[0] == "nguyen-hue"


def test_l3_candidate_from_le_loi_on_pasteur(view: GraphView) -> None:
    candidates = {
        (candidate.slots["cross_street"], candidate.slots["oneway_street"]): candidate
        for candidate in l3_next_on_oneway_candidates(view)
    }
    candidate = candidates[("Lê Lợi", "Pasteur")]

    assert candidate.answer == StreetNameAnswer(street_name="Hai Bà Trưng")
    assert candidate.features.hop_count == 1
    assert candidate.features.depends_on_edge_direction


def test_l5_common_neighbours(view: GraphView) -> None:
    candidates = {
        (candidate.slots["street_a"], candidate.slots["street_b"]): candidate
        for candidate in l5_common_neighbours_candidates(view)
    }

    # Le Loi and Hai Ba Trung do not meet; Pasteur and "Duong 10A" meet both.
    assert candidates[("Hai Bà Trưng", "Lê Lợi")].answer.model_dump()["value"] == 2.0


def test_slot_name_drops_the_street_prefix(view: GraphView) -> None:

    assert slot_name(view, "10a") == "10A"
    assert slot_name(view, "pasteur") == "Pasteur"


def test_shortest_route_respects_direction(view: GraphView) -> None:

    route = shortest_route(view, {1}, {7})
    assert route is not None
    assert route.length_m == pytest.approx(300.0)
    assert route.nodes == [1, 2, 3, 7]
    assert route_street_sequence(view, route.edges) == ["Pasteur", "Hai Bà Trưng"]

    against_direction = shortest_route(view, {8}, {6})
    assert against_direction is not None
    assert against_direction.nodes == [8, 1, 2, 3, 6]
    assert shortest_route(view, {3}, {1}) is None
    assert shortest_length_ignoring_direction(view, {3}, {1}) == pytest.approx(200.0)


def test_route_through_a_one_way_street_in_its_direction_only(view: GraphView) -> None:
    stages = [{"hai-ba-trung"}, {"10a"}, {"le-loi"}]

    assert shortest_length_following_sequence(view, {3}, {5}, stages) == pytest.approx(400.0)
    reverse_stages = [{"le-loi"}, {"10a"}, {"hai-ba-trung"}]
    assert shortest_length_following_sequence(view, {5}, {3}, reverse_stages) is None
    assert shortest_length_following_sequence(
        view, {5}, {3}, reverse_stages, respect_rules=False
    ) == pytest.approx(400.0)


def test_sampling_balances_strata(view: GraphView) -> None:
    candidates = l3_next_on_oneway_candidates(view)

    assert len(sample_candidates(candidates, 1, random.Random(0))) == 1
    with pytest.raises(ValueError):
        sample_candidates(candidates, 100, random.Random(0))


def test_street_length_halves_two_way_streets(view: GraphView) -> None:

    assert view.street_length_m("pasteur") == pytest.approx(200.0)
    assert view.street_length_m("le-loi") == pytest.approx(300.0)


def test_length_questions_keep_a_margin_and_balance_answers(view: GraphView) -> None:
    candidates = l1_length_candidates(view, random.Random(0))

    for candidate in candidates:
        street = candidate.slots["street"]
        street_id = next(
            s.street_id for s in view.streets.values() if slot_name(view, s.street_id) == street
        )
        length_m = view.street_length_m(street_id)
        threshold_m = float(candidate.slots["threshold_m"])
        assert abs(length_m - threshold_m) >= 0.25 * length_m
        assert candidate.answer.model_dump()["value"] == (length_m > threshold_m)
