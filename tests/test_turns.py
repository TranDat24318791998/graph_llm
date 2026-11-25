import networkx as nx

from graph_llm.graph.types import RoadGraph
from graph_llm.questions.turns import TurnRules, shortest_route_with_turns
from graph_llm.schemas.question import EdgeRef


def edge(source: int, target: int) -> EdgeRef:
    return EdgeRef(source_node_id=source, target_node_id=target, key=0)


def grid() -> RoadGraph:
    """Two-way grid:

        1 --- 2 --- 3
        |     |     |
        4 --- 5 --- 6

    Horizontal edges 100 m, vertical edges 100 m."""
    graph: RoadGraph = nx.MultiDiGraph()
    for first, second in ((1, 2), (2, 3), (4, 5), (5, 6), (1, 4), (2, 5), (3, 6)):
        graph.add_edge(first, second, length=100.0)
        graph.add_edge(second, first, length=100.0)
    return graph


def test_without_rules_the_route_is_the_plain_shortest_path() -> None:
    route = shortest_route_with_turns(grid(), [1], [6], TurnRules([]))

    assert route is not None
    assert route.length_m == 300.0


def test_a_forbidden_left_turn_forces_a_detour() -> None:
    """From 1 to 5 the routes 1-2-5 and 1-4-5 both take 200 m. Forbid the turn 1->2->5
    and the turn 1->4->5: the shortest legal route must avoid both and is longer."""
    rules = TurnRules([(edge(1, 2), edge(2, 5)), (edge(1, 4), edge(4, 5))])

    route = shortest_route_with_turns(grid(), [1], [5], rules)

    assert route is not None
    assert route.length_m == 400.0
    assert not rules.route_violates(route.edges)


def test_a_three_edge_sequence_is_enforced() -> None:
    """Forbid 2->3, 3->6, 6->5 (a U-turn-like sequence through a via way 3-6)."""
    rules = TurnRules([(edge(2, 3), edge(3, 6), edge(6, 5))])

    assert rules.route_violates([edge(1, 2), edge(2, 3), edge(3, 6), edge(6, 5)])
    assert not rules.route_violates([edge(2, 3), edge(3, 6)])


def test_no_legal_route() -> None:
    graph: RoadGraph = nx.MultiDiGraph()
    graph.add_edge(1, 2, length=10.0)
    graph.add_edge(2, 3, length=10.0)
    rules = TurnRules([(edge(1, 2), edge(2, 3))])

    assert shortest_route_with_turns(graph, [1], [3], rules) is None
