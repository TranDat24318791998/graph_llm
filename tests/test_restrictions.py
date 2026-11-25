import networkx as nx

from graph_llm.graph.restrictions import MappingStatus, map_restriction, parse_restriction
from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.question import EdgeRef


def element(tags: dict[str, str], members: list[tuple[str, int, str]]) -> dict[str, object]:
    return {
        "id": 7,
        "tags": {"type": "restriction", **tags},
        "members": [{"type": t, "ref": r, "role": role} for t, r, role in members],
    }


def test_parse_plain_and_conditional_restrictions() -> None:
    plain = parse_restriction(
        element(
            {"restriction": "no_left_turn", "except": "bicycle;motorcycle"},
            [("way", 1, "from"), ("node", 5, "via"), ("way", 2, "to")],
        )
    )
    conditional = parse_restriction(
        element(
            {"restriction:conditional": "no_right_turn @ (06:00-22:00)"},
            [("way", 1, "from"), ("node", 5, "via"), ("way", 3, "to")],
        )
    )

    assert plain is not None and conditional is not None
    assert plain.kind == "no_left_turn" and plain.is_unconditional and plain.applies_to_cars
    assert plain.exempt_vehicles == ["bicycle", "motorcycle"]
    assert conditional.kind == "no_right_turn" and conditional.condition == "06:00-22:00"
    assert not conditional.is_unconditional


def test_restriction_for_motorcycles_only_does_not_apply_to_cars() -> None:
    restriction = parse_restriction(
        element(
            {"restriction:motorcycle": "no_u_turn"},
            [("way", 1, "from"), ("node", 5, "via"), ("way", 1, "to")],
        )
    )

    assert restriction is not None
    assert not restriction.applies_to_cars


def crossing() -> RoadGraph:
    """Node 5 is a crossing: way 1 comes from the west (4 -> 5), way 2 goes north (5 -> 6),
    way 3 goes south (5 -> 7), way 1 continues east (5 -> 8)."""
    graph: RoadGraph = nx.MultiDiGraph()
    for source, target, way in ((4, 5, 1), (5, 6, 2), (5, 7, 3), (5, 8, 1)):
        graph.add_edge(source, target, osmid=way, length=100.0)
    return graph


def test_prohibitory_restriction_forbids_one_turn() -> None:
    restriction = parse_restriction(
        element(
            {"restriction": "no_left_turn"},
            [("way", 1, "from"), ("node", 5, "via"), ("way", 2, "to")],
        )
    )
    assert restriction is not None

    mapped = map_restriction(crossing(), restriction)

    assert mapped is not None
    into = EdgeRef(source_node_id=4, target_node_id=5, key=0)
    north = EdgeRef(source_node_id=5, target_node_id=6, key=0)
    assert mapped.forbidden_sequences == frozenset({(into, north)})


def test_mandatory_restriction_forbids_every_other_exit() -> None:
    restriction = parse_restriction(
        element(
            {"restriction": "only_straight_on"},
            [("way", 1, "from"), ("node", 5, "via"), ("way", 1, "to")],
        )
    )
    assert restriction is not None

    mapped = map_restriction(crossing(), restriction)

    assert mapped is not None
    into = EdgeRef(source_node_id=4, target_node_id=5, key=0)
    exits = {turn[1].target_node_id for turn in mapped.forbidden_sequences if turn[0] == into}
    assert exits == {6, 7}


def test_via_way_restriction_is_not_mapped() -> None:
    restriction = parse_restriction(
        element(
            {"restriction": "no_u_turn"},
            [("way", 1, "from"), ("way", 9, "via"), ("way", 1, "to")],
        )
    )
    assert restriction is not None

    assert map_restriction(crossing(), restriction).status is MappingStatus.OUTSIDE_GRAPH


def edge(source: int, target: int) -> EdgeRef:
    return EdgeRef(source_node_id=source, target_node_id=target, key=0)


def test_u_turn_through_a_median_crossing_way() -> None:
    """Divided road: eastbound 1 -> 2 (way 10), westbound 3 -> 4 (way 20); the median
    crossing 2 -> 3 is way 30. A no_u_turn from way 10 via way 30 to way 20 forbids
    the sequence 1->2, 2->3, 3->4."""
    graph: RoadGraph = nx.MultiDiGraph()
    for source, target, way in ((1, 2, 10), (3, 4, 20), (2, 3, 30), (2, 5, 10)):
        graph.add_edge(source, target, osmid=way, length=50.0)
    restriction = parse_restriction(
        element(
            {"restriction": "no_u_turn"},
            [("way", 10, "from"), ("way", 30, "via"), ("way", 20, "to")],
        )
    )
    assert restriction is not None

    mapped = map_restriction(graph, restriction)

    assert mapped.status is MappingStatus.MAPPED
    assert mapped.forbidden_sequences == frozenset({(edge(1, 2), edge(2, 3), edge(3, 4))})


def test_turn_into_a_one_way_street_against_its_direction_is_already_impossible() -> None:
    graph = crossing()
    graph.remove_edge(5, 6)
    graph.add_edge(6, 5, osmid=2, length=100.0)
    restriction = parse_restriction(
        element(
            {"restriction": "no_left_turn"},
            [("way", 1, "from"), ("node", 5, "via"), ("way", 2, "to")],
        )
    )
    assert restriction is not None

    assert map_restriction(graph, restriction).status is MappingStatus.ALREADY_IMPOSSIBLE
