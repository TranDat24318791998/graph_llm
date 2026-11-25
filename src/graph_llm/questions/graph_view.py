"""Read-only view of a processed graph with the street-level facts that questions use.

All street-level facts are defined here once, so question generation, answer
computation, text generation and scoring agree on them. Graph v2 definitions
(docs/graph_v2_design.md):

- Two streets meet when they reach the same junction (``graph/junctions.py``).
- A street's direction is two-way, two-way divided, one-way or mixed, from its
  edge labels (``graph/carriageways.py``).
- A street's length is its centre-line length.
- Places (named squares and roundabouts) are junction names, not streets.
- Routes obey the unconditional car turn restrictions; time-conditional ones are
  kept separately so that questions can avoid them.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from graph_llm.graph.carriageways import (
    EdgeDirection,
    StreetDirection,
    centre_line_length_m,
    classify_all_streets,
    street_direction,
)
from graph_llm.graph.junctions import build_junctions, place_street_ids
from graph_llm.graph.reference_graph import (
    ReferenceGraph,
    excluded_item_ids,
    load_reference_graph,
    load_review_log,
)
from graph_llm.graph.restrictions import MappingStatus, map_restriction
from graph_llm.graph.types import RoadGraph
from graph_llm.questions.turns import TurnRules
from graph_llm.schemas.junction import Junction, JunctionSettings
from graph_llm.schemas.question import EdgeRef
from graph_llm.schemas.restriction import TurnRestriction
from graph_llm.schemas.street import Street, StreetCategory

# Nodes of one intersection lie at most this far apart; streets that meet at a wider
# junction, or at several junctions, have no single "intersection of A and B".
INTERSECTION_SPREAD_LIMIT_M = 80.0

# Share of a street's length that must carry a lanes value to state its lane count.
LANES_COVERAGE_THRESHOLD = 0.9


def haversine_m(
    longitude_a: float, latitude_a: float, longitude_b: float, latitude_b: float
) -> float:
    """Return the great-circle distance between two points in metres."""
    earth_radius_m = 6_371_008.8
    phi_a, phi_b = math.radians(latitude_a), math.radians(latitude_b)
    delta_phi = phi_b - phi_a
    delta_lambda = math.radians(longitude_b - longitude_a)
    half_chord = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * earth_radius_m * math.asin(math.sqrt(half_chord))


def car_turn_rules(
    graph: RoadGraph, restrictions: Sequence[TurnRestriction]
) -> tuple[TurnRules, TurnRules]:
    """Return the forbidden edge sequences for cars: (always applying, time-conditional)."""
    always = TurnRules([])
    timed = TurnRules([])
    for restriction in restrictions:
        if not restriction.applies_to_cars:
            continue
        mapped = map_restriction(graph, restriction)
        if mapped.status is not MappingStatus.MAPPED:
            continue
        rules = always if restriction.is_unconditional else timed
        for sequence in mapped.forbidden_sequences:
            rules.add(sequence)
    return always, timed


@dataclass(frozen=True)
class GraphView:
    """A processed graph with lookups between edges, nodes, streets and junctions."""

    graph: RoadGraph
    streets: dict[str, Street]
    street_ids_of_edge: dict[EdgeRef, list[str]]
    nodes_of_street: dict[str, set[int]]
    street_ids_at_node: dict[int, set[str]]
    edge_labels: dict[str, dict[EdgeRef, EdgeDirection]]
    junctions: dict[str, Junction]
    junction_id_of_node: dict[int, str]
    junction_ids_of_street: dict[str, set[str]]
    place_ids: frozenset[str]
    excluded_ids: frozenset[str]
    always_rules: TurnRules
    timed_rules: TurnRules

    @classmethod
    def build(
        cls,
        graph: RoadGraph,
        street_list: list[Street],
        settings: JunctionSettings | None = None,
        restrictions: Sequence[TurnRestriction] = (),
        junction_list: list[Junction] | None = None,
        excluded_ids: frozenset[str] = frozenset(),
    ) -> GraphView:
        """Build every lookup; junctions are computed unless a frozen list is given."""
        settings = settings or JunctionSettings()
        streets = {street.street_id: street for street in street_list}
        street_ids_of_edge: dict[EdgeRef, list[str]] = defaultdict(list)
        nodes_of_street: dict[str, set[int]] = defaultdict(set)
        street_ids_at_node: dict[int, set[str]] = defaultdict(set)
        for street in street_list:
            for edge in street.edges:
                street_ids_of_edge[edge].append(street.street_id)
                for node in (edge.source_node_id, edge.target_node_id):
                    nodes_of_street[street.street_id].add(node)
                    street_ids_at_node[node].add(street.street_id)
        labels = classify_all_streets(graph, street_list, settings.carriageway_max_separation_m)
        if junction_list is None:
            junction_list = build_junctions(graph, street_list, labels, settings)
        junctions = {junction.junction_id: junction for junction in junction_list}
        junction_id_of_node = {n: j.junction_id for j in junction_list for n in j.node_ids}
        junction_ids_of_street: dict[str, set[str]] = defaultdict(set)
        for junction in junction_list:
            for street_id in junction.street_ids:
                junction_ids_of_street[street_id].add(junction.junction_id)
        always, timed = car_turn_rules(graph, restrictions)
        return cls(
            graph=graph,
            streets=streets,
            street_ids_of_edge=dict(street_ids_of_edge),
            nodes_of_street=dict(nodes_of_street),
            street_ids_at_node=dict(street_ids_at_node),
            edge_labels=labels,
            junctions=junctions,
            junction_id_of_node=junction_id_of_node,
            junction_ids_of_street=dict(junction_ids_of_street),
            place_ids=frozenset(place_street_ids(street_list, settings)),
            excluded_ids=excluded_ids,
            always_rules=always,
            timed_rules=timed,
        )

    @classmethod
    def from_reference(
        cls, reference: ReferenceGraph, excluded_ids: frozenset[str] = frozenset()
    ) -> GraphView:
        """Build the view of a frozen reference graph, with its frozen junctions."""
        return cls.build(
            reference.graph,
            reference.streets,
            reference.metadata.config.junctions,
            reference.restrictions,
            reference.junctions,
            excluded_ids,
        )

    @classmethod
    def from_graph_and_streets(cls, graph: RoadGraph, street_list: list[Street]) -> GraphView:
        """Build a view with default settings and no turn restrictions (used by tests)."""
        return cls.build(graph, street_list)

    def edge_attributes(self, edge: EdgeRef) -> dict[str, object]:
        """Return the attributes of one edge."""
        attributes: dict[str, object] = self.graph.edges[
            edge.source_node_id, edge.target_node_id, edge.key
        ]
        return attributes

    def edge_length_m(self, edge: EdgeRef) -> float:
        """Return the length of one edge in metres."""
        return float(self.edge_attributes(edge)["length"])  # type: ignore[arg-type]

    def is_place(self, street_id: str) -> bool:
        """Return True if the named item is a square or roundabout, not a street."""
        return street_id in self.place_ids

    def is_regular_street(self, street_id: str) -> bool:
        """Return True for a regular street that is not a place."""
        return self.streets[street_id].category is StreetCategory.REGULAR and not self.is_place(
            street_id
        )

    def is_askable(self, street_id: str) -> bool:
        """Return True if Phase 1 questions may name this street."""
        street = self.streets[street_id]
        return (
            self.is_regular_street(street_id)
            and not street.shares_name_with_other_street
            and street_id not in self.excluded_ids
        )

    def regular_street_ids_at(self, node: int) -> set[str]:
        """Return the regular streets that reach the junction of a node (or the node)."""
        junction_id = self.junction_id_of_node.get(node)
        if junction_id is None:
            candidates = self.street_ids_at_node.get(node, set())
        else:
            candidates = set(self.junctions[junction_id].street_ids)
        return {street_id for street_id in candidates if self.is_regular_street(street_id)}

    def regular_neighbours(self, street_id: str) -> dict[str, set[int]]:
        """Return the other regular streets that meet a street, with the junction nodes."""
        neighbours: dict[str, set[int]] = defaultdict(set)
        for junction_id in self.junction_ids_of_street.get(street_id, set()):
            junction = self.junctions[junction_id]
            for other_id in junction.street_ids:
                if other_id != street_id and self.is_regular_street(other_id):
                    neighbours[other_id] |= set(junction.node_ids)
        return dict(neighbours)

    def incident_street_edges(self, node: int, street_id: str) -> list[EdgeRef]:
        """Return the edges of a street that start or end at a node."""
        return [
            edge
            for edge in self.streets[street_id].edges
            if node in (edge.source_node_id, edge.target_node_id)
        ]

    def node_spread_m(self, nodes: set[int]) -> float:
        """Return the largest distance between two nodes of a set."""
        coordinates = [
            (float(self.graph.nodes[node]["x"]), float(self.graph.nodes[node]["y"]))
            for node in nodes
        ]
        return max(
            (
                haversine_m(*first, *second)
                for index, first in enumerate(coordinates)
                for second in coordinates[index + 1 :]
            ),
            default=0.0,
        )

    def meeting_junctions(self, first_id: str, second_id: str) -> list[Junction]:
        """Return the junctions where two streets meet."""
        shared = self.junction_ids_of_street.get(first_id, set()) & self.junction_ids_of_street.get(
            second_id, set()
        )
        return [self.junctions[junction_id] for junction_id in sorted(shared)]

    def clean_intersection(self, first_id: str, second_id: str) -> set[int] | None:
        """Return the junction nodes where two streets meet, if that place is unique.

        Unique means the streets meet at exactly one junction, and that junction is at
        most ``INTERSECTION_SPREAD_LIMIT_M`` wide.
        """
        junctions = self.meeting_junctions(first_id, second_id)
        if len(junctions) != 1 or junctions[0].spread_m > INTERSECTION_SPREAD_LIMIT_M:
            return None
        return set(junctions[0].node_ids)

    def direction(self, street_id: str) -> StreetDirection:
        """Return the direction status of a street."""
        return street_direction(self.edge_labels[street_id])

    def lane_count(self, street_id: str) -> int | None:
        """Return the lane count of a street, if OSM states one value for nearly all of it."""
        total_length_m = 0.0
        tagged_length_m = 0.0
        values: set[int] = set()
        for edge in self.streets[street_id].edges:
            length_m = self.edge_length_m(edge)
            total_length_m += length_m
            raw_value = self.edge_attributes(edge).get("lanes")
            if raw_value is None or isinstance(raw_value, list):
                continue
            try:
                values.add(int(str(raw_value)))
            except ValueError:
                continue
            tagged_length_m += length_m
        if len(values) != 1 or tagged_length_m < LANES_COVERAGE_THRESHOLD * total_length_m:
            return None
        return values.pop()

    def street_length_m(self, street_id: str) -> float:
        """Return the length of a street along its centre line (each stretch counted once)."""
        labels = self.edge_labels[street_id]
        return centre_line_length_m(labels, {edge: self.edge_length_m(edge) for edge in labels})

    def shortest_edge(self, source: int, target: int) -> EdgeRef:
        """Return the shortest of the parallel edges from source to target."""
        parallel = self.graph.get_edge_data(source, target)
        key = min(parallel, key=lambda edge_key: float(parallel[edge_key]["length"]))
        return EdgeRef(source_node_id=source, target_node_id=target, key=key)


def load_graph_view(
    processed_data_directory: Path, raw_data_directory: Path, graph_version: str
) -> GraphView:
    """Load a frozen reference graph and apply the exclusions of its review log."""
    reference = load_reference_graph(processed_data_directory, raw_data_directory, graph_version)
    review_path = reference.metadata.config.review_log_path
    review_log = load_review_log(Path(review_path)) if review_path else None
    return GraphView.from_reference(reference, excluded_item_ids(review_log))
