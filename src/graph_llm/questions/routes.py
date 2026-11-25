"""Shortest legal routes and routes that follow a street sequence (graph v2).

Used twice: to compute route ground truth and to score predicted routes. Legal
means: edges are used in their direction, and no unconditional car turn
restriction is broken (``GraphView.always_rules``). Unnamed edges and edges of
places (named squares and roundabouts) belong to junctions, so a street sequence
never names them.
"""

from __future__ import annotations

import heapq
import itertools

import networkx as nx

from graph_llm.questions.graph_view import GraphView
from graph_llm.questions.turns import TurnAwareRoute, shortest_route_with_turns
from graph_llm.schemas.question import EdgeRef


def shortest_route(view: GraphView, sources: set[int], targets: set[int]) -> TurnAwareRoute | None:
    """Return the shortest legal route from any source to any target."""
    return shortest_route_with_turns(view.graph, sources, targets, view.always_rules)


def shortest_length_ignoring_direction(
    view: GraphView, sources: set[int], targets: set[int]
) -> float | None:
    """Return the shortest distance if edge directions and turn restrictions are ignored."""
    undirected = view.graph.to_undirected(as_view=True)
    lengths: dict[int, float]
    lengths, _ = nx.multi_source_dijkstra(undirected, sources, weight="length")  # type: ignore[assignment]
    reachable = [lengths[target] for target in targets if target in lengths]
    return float(min(reachable)) if reachable else None


def counted_owners(view: GraphView, edge: EdgeRef) -> list[str]:
    """Return the streets an edge counts for in a street sequence (places excluded)."""
    return [o for o in view.street_ids_of_edge.get(edge, []) if not view.is_place(o)]


def route_street_sequence(view: GraphView, edges: list[EdgeRef]) -> list[str] | None:
    """Return the street names along a route, without repeats, unnamed and place edges.

    Returns:
        The display names in order, or None if the route uses an alley or internal
        road, or an edge of several streets, because such a route has no unambiguous
        sequence of regular street names.
    """
    street_ids: list[str] = []
    for edge in edges:
        owners = counted_owners(view, edge)
        if not owners:
            continue
        if len(owners) > 1 or not view.is_regular_street(owners[0]):
            return None
        if not street_ids or street_ids[-1] != owners[0]:
            street_ids.append(owners[0])
    return [view.streets[street_id].display_name for street_id in street_ids]


def shortest_length_following_sequence(
    view: GraphView,
    sources: set[int],
    targets: set[int],
    stages: list[set[str]],
    respect_rules: bool = True,
) -> float | None:
    """Return the length of the shortest route that uses the streets of each stage in order.

    Within a stage the route may use edges of the stage's streets, unnamed edges and
    place edges. It moves to the next stage by taking an edge of the next stage's
    streets. With ``respect_rules`` the route is legal (directions and unconditional
    turn restrictions); without it, edges may be used in both directions and turn
    restrictions are ignored, which tells a wrong-direction route from a wrong one.

    Args:
        view: The graph.
        sources: Start nodes.
        targets: End nodes.
        stages: For each street in the sequence, the street ids it may stand for.
        respect_rules: Whether directions and turn restrictions apply.

    Returns:
        The length in metres, or None if no such route exists.
    """
    if not stages:
        return None
    outgoing: dict[int, list[tuple[EdgeRef, int, float, set[str]]]] = {}
    for source, target, key, attributes in view.graph.edges(keys=True, data=True):
        edge = EdgeRef(source_node_id=source, target_node_id=target, key=key)
        owners = set(counted_owners(view, edge))
        length_m = float(attributes["length"])
        outgoing.setdefault(source, []).append((edge, target, length_m, owners))
        if not respect_rules:
            outgoing.setdefault(target, []).append((edge, source, length_m, owners))

    rules = view.always_rules
    final_stage = len(stages) - 1
    counter = itertools.count()
    # State: (node reached, stage, matched restriction prefixes).
    start_state: frozenset[int] = frozenset()
    queue: list[tuple[float, int, int, int, frozenset[int]]] = [
        (0.0, next(counter), node, 0, start_state) for node in sources
    ]
    settled: set[tuple[int, int, frozenset[int]]] = set()
    while queue:
        distance, _, node, stage, active = heapq.heappop(queue)
        if (node, stage, active) in settled:
            continue
        settled.add((node, stage, active))
        if stage == final_stage and node in targets:
            return distance
        for edge, next_node, length_m, owners in outgoing.get(node, []):
            next_active: frozenset[int] | None = frozenset()
            if respect_rules:
                next_active = rules.step(active, edge)
                if next_active is None:
                    continue
            assert next_active is not None
            if not owners or owners & stages[stage]:
                heapq.heappush(
                    queue, (distance + length_m, next(counter), next_node, stage, next_active)
                )
            if stage < final_stage and owners & stages[stage + 1]:
                heapq.heappush(
                    queue,
                    (distance + length_m, next(counter), next_node, stage + 1, next_active),
                )
    return None
