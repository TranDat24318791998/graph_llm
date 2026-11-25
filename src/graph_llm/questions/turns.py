"""Shortest routes that obey turn restrictions, and checks of routes against them.

A turn restriction forbids a sequence of consecutive edges: (in-edge, out-edge) at
a via node, or (in-edge, via edges..., out-edge) through a via way. Node-based
Dijkstra cannot enforce this, so the search runs over states: the edge just
travelled plus the restriction prefixes that the route currently matches. The
forbidden sequences are stored in a prefix tree (trie); stepping onto an edge
that completes a sequence is not allowed.

The graph itself is unchanged; restrictions only remove some transitions.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.question import EdgeRef


@dataclass
class TrieNode:
    """One prefix of the forbidden sequences."""

    children: dict[EdgeRef, int] = field(default_factory=dict)
    terminal: bool = False


class TurnRules:
    """A set of forbidden edge sequences, ready for matching while routing."""

    def __init__(self, sequences: Iterable[tuple[EdgeRef, ...]]) -> None:
        """Build the prefix tree of the forbidden sequences."""
        self.nodes: list[TrieNode] = [TrieNode()]
        self.sequence_count = 0
        for sequence in sequences:
            self.add(sequence)

    def add(self, sequence: tuple[EdgeRef, ...]) -> None:
        """Add one forbidden sequence of at least two edges."""
        if len(sequence) < 2:
            raise ValueError("A forbidden sequence needs at least two edges.")
        current = 0
        for edge in sequence:
            child = self.nodes[current].children.get(edge)
            if child is None:
                child = len(self.nodes)
                self.nodes.append(TrieNode())
                self.nodes[current].children[edge] = child
            current = child
        self.nodes[current].terminal = True
        self.sequence_count += 1

    def step(self, active: frozenset[int], edge: EdgeRef) -> frozenset[int] | None:
        """Return the prefixes matched after travelling ``edge``; None if a sequence completes.

        Every edge may also start a new sequence, so the root is always active.
        """
        matched = set()
        for prefix in active | {0}:
            child = self.nodes[prefix].children.get(edge)
            if child is None:
                continue
            if self.nodes[child].terminal:
                return None
            matched.add(child)
        return frozenset(matched)

    def route_violates(self, edges: Sequence[EdgeRef]) -> bool:
        """Return True if the route contains a forbidden sequence."""
        active: frozenset[int] = frozenset()
        for edge in edges:
            stepped = self.step(active, edge)
            if stepped is None:
                return True
            active = stepped
        return False


@dataclass(frozen=True)
class TurnAwareRoute:
    """A shortest route that obeys the rules: its edges and length."""

    edges: list[EdgeRef]
    length_m: float

    @property
    def nodes(self) -> list[int]:
        """Return the node sequence of the route."""
        if not self.edges:
            return []
        return [self.edges[0].source_node_id] + [edge.target_node_id for edge in self.edges]


def shortest_route_with_turns(
    graph: RoadGraph,
    sources: Iterable[int],
    targets: Iterable[int],
    rules: TurnRules,
) -> TurnAwareRoute | None:
    """Return the shortest route from any source to any target that obeys the rules.

    Returns None if no legal route exists. A route of zero edges is returned when a
    source is also a target.
    """
    target_set = set(targets)
    source_list = list(sources)
    if target_set & set(source_list):
        return TurnAwareRoute(edges=[], length_m=0.0)
    counter = itertools.count()
    queue: list[tuple[float, int, EdgeRef, frozenset[int], tuple[EdgeRef, ...]]] = []
    for source in source_list:
        for _, target, key, data in graph.out_edges(source, keys=True, data=True):
            edge = EdgeRef(source_node_id=source, target_node_id=target, key=key)
            active = rules.step(frozenset(), edge)
            if active is None:
                continue
            heapq.heappush(queue, (float(data["length"]), next(counter), edge, active, (edge,)))
    settled: set[tuple[EdgeRef, frozenset[int]]] = set()
    while queue:
        length, _, edge, active, path = heapq.heappop(queue)
        state = (edge, active)
        if state in settled:
            continue
        settled.add(state)
        node = edge.target_node_id
        if node in target_set:
            return TurnAwareRoute(edges=list(path), length_m=length)
        for _, target, key, data in graph.out_edges(node, keys=True, data=True):
            following = EdgeRef(source_node_id=node, target_node_id=target, key=key)
            stepped = rules.step(active, following)
            if stepped is None or (following, stepped) in settled:
                continue
            heapq.heappush(
                queue,
                (
                    length + float(data["length"]),
                    next(counter),
                    following,
                    stepped,
                    path + (following,),
                ),
            )
    return None
