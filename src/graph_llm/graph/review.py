"""Build the review cards of a processed graph: one per regular street and junction.

Each card states the computed facts and the automatic flags that need a human
look (docs/graph_v2_design.md, Section 4). Flags do not mean an error; they mark
cases where a rule may not match reality. Decisions are recorded separately in a
review log.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx

from graph_llm.graph.carriageways import (
    EdgeDirection,
    StreetEdgeLabels,
    centre_line_length_m,
    edge_polyline,
    midpoint_and_heading,
    street_direction,
)
from graph_llm.graph.restrictions import MappingStatus, map_restriction, way_ids_of_edge
from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.graph import BoundingBox
from graph_llm.schemas.junction import Junction, JunctionKind
from graph_llm.schemas.question import EdgeRef
from graph_llm.schemas.restriction import TurnRestriction
from graph_llm.schemas.review import JunctionCard, ReviewFlag, StreetCard
from graph_llm.schemas.street import Street, StreetCategory

METRES_PER_DEGREE = 111_320.0
BOUNDARY_MARGIN_M = 60.0
SHORT_STREET_M = 60.0
DIVIDED_BALANCE_MIN = 0.7
WIDE_JUNCTION_M = 60.0
MANY_NODES = 6
CONNECTOR_SPREAD_M = 30.0


@dataclass(frozen=True)
class ReviewInputs:
    """Everything the review cards are computed from."""

    graph: RoadGraph
    streets: Sequence[Street]
    labels: StreetEdgeLabels
    junctions: Sequence[Junction]
    restrictions: Sequence[TurnRestriction]
    area: BoundingBox


def distance_to_boundary_m(graph: RoadGraph, node: int, area: BoundingBox) -> float:
    """Return the distance in metres from a node to the edge of the study rectangle."""
    x, y = float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])
    scale = math.cos(math.radians(y)) * METRES_PER_DEGREE
    return min(
        (x - area.west) * scale,
        (area.east - x) * scale,
        (y - area.south) * METRES_PER_DEGREE,
        (area.north - y) * METRES_PER_DEGREE,
    )


def street_axis(graph: RoadGraph, edges: Sequence[EdgeRef]) -> float:
    """Return the main direction of a street (an angle modulo pi), length-weighted."""
    total = complex(0.0, 0.0)
    for edge in edges:
        _, heading = midpoint_and_heading(edge_polyline(graph, edge))
        length = float(graph.edges[edge.source_node_id, edge.target_node_id, edge.key]["length"])
        total += length * complex(math.cos(2 * heading), math.sin(2 * heading))
    return math.atan2(total.imag, total.real) / 2


def street_card(
    inputs: ReviewInputs,
    street: Street,
    junction_ids_of_street: Mapping[str, list[str]],
    junction_by_id: Mapping[str, Junction],
    names: Mapping[str, str],
    junction_of_node: Mapping[int, str],
) -> StreetCard:
    """Return the review card of one regular street."""
    graph = inputs.graph
    labels = dict(inputs.labels[street.street_id])
    lengths = {
        edge: float(graph.edges[edge.source_node_id, edge.target_node_id, edge.key]["length"])
        for edge in street.edges
    }
    by_label: dict[str, float] = defaultdict(float)
    for edge, label in labels.items():
        by_label[label.value] += lengths[edge]
    direction = street_direction(labels)
    flags: list[ReviewFlag] = []
    if direction.value == "mixed":
        one_way_share = by_label["one_way"] / max(1.0, sum(by_label.values()))
        flags.append(
            ReviewFlag(
                code="S-MIXED",
                message=(
                    f"two-way {by_label['two_way'] / 2:.0f} m, divided "
                    f"{by_label['divided'] / 2:.0f} m, one-way {by_label['one_way']:.0f} m "
                    f"(one-way share of edge length {one_way_share:.0%})"
                ),
            )
        )
    divided = [edge for edge, label in labels.items() if label is EdgeDirection.DIVIDED]
    if divided:
        axis = street_axis(graph, list(street.edges))
        forward = sum(
            lengths[e]
            for e in divided
            if math.cos(midpoint_and_heading(edge_polyline(graph, e))[1] - axis) >= 0
        )
        backward = sum(lengths[e] for e in divided) - forward
        balance = min(forward, backward) / max(forward, backward, 1.0)
        if balance < DIVIDED_BALANCE_MIN:
            flags.append(
                ReviewFlag(
                    code="S-DIVIDED-IMBALANCE",
                    message=f"carriageway lengths {forward:.0f} m and {backward:.0f} m",
                )
            )
    undirected: nx.Graph[int] = nx.Graph()
    undirected.add_edges_from((e.source_node_id, e.target_node_id) for e in street.edges)
    # Nodes of one junction count as one point: a street that enters a roundabout at one
    # node and leaves it at another is still one piece.
    contracted: nx.Graph[str] = nx.Graph()
    for e in street.edges:
        contracted.add_edge(
            junction_of_node.get(e.source_node_id, f"n{e.source_node_id}"),
            junction_of_node.get(e.target_node_id, f"n{e.target_node_id}"),
        )
    pieces = nx.number_connected_components(contracted)
    if pieces > 1:
        flags.append(ReviewFlag(code="S-PIECES", message=f"{pieces} separate pieces"))
    inner_dead_ends = sorted(
        node
        for node in undirected.nodes
        if len(set(graph.successors(node)) | set(graph.predecessors(node))) == 1
        and distance_to_boundary_m(graph, node, inputs.area) > BOUNDARY_MARGIN_M
    )
    if inner_dead_ends:
        flags.append(
            ReviewFlag(
                code="S-INNER-DEAD-END",
                message=f"dead end inside the area at nodes {inner_dead_ends}",
            )
        )
    touches_boundary = any(
        distance_to_boundary_m(graph, node, inputs.area) <= BOUNDARY_MARGIN_M
        for node in undirected.nodes
    )
    centre_m = centre_line_length_m(labels, lengths)
    if centre_m < SHORT_STREET_M:
        flags.append(ReviewFlag(code="S-SHORT", message=f"centre line {centre_m:.0f} m"))
    if street.shares_name_with_other_street:
        flags.append(ReviewFlag(code="S-SHARED-NAME", message="another street has the same name"))
    multi_named = sum(
        isinstance(graph.edges[e.source_node_id, e.target_node_id, e.key].get("name"), list)
        for e in street.edges
    )
    if multi_named:
        flags.append(
            ReviewFlag(
                code="S-MULTI-NAME-EDGES", message=f"{multi_named} edges carry several names"
            )
        )
    junction_ids = junction_ids_of_street.get(street.street_id, [])
    meeting = sorted(
        {
            names[other]
            for junction_id in junction_ids
            for other in junction_by_id[junction_id].street_ids
            if other != street.street_id and other in names
        }
    )
    if not meeting:
        flags.append(ReviewFlag(code="S-ISOLATED", message="meets no other regular street"))
    places = sorted({name for jid in junction_ids for name in junction_by_id[jid].place_names})
    nodes = list(undirected.nodes)
    return StreetCard(
        street_id=street.street_id,
        name=street.display_name,
        direction=direction.value,
        centre_line_length_m=round(centre_m, 1),
        two_way_length_m=round(by_label["two_way"] / 2, 1),
        divided_length_m=round(by_label["divided"] / 2, 1),
        one_way_length_m=round(by_label["one_way"], 1),
        junction_ids=junction_ids,
        meeting_street_names=meeting,
        places=places,
        inner_dead_end_node_ids=inner_dead_ends,
        touches_area_boundary=touches_boundary,
        latitude=sum(float(graph.nodes[n]["y"]) for n in nodes) / len(nodes),
        longitude=sum(float(graph.nodes[n]["x"]) for n in nodes) / len(nodes),
        flags=flags,
    )


def restriction_nodes(graph: RoadGraph, restriction: TurnRestriction) -> set[int]:
    """Return the graph nodes a restriction acts at (its via node, or its via way's nodes)."""
    if restriction.via_node_ids:
        return {n for n in restriction.via_node_ids if n in graph}
    via = set(restriction.via_way_ids)
    nodes: set[int] = set()
    for source, target, key in graph.edges(keys=True):
        edge = EdgeRef(source_node_id=source, target_node_id=target, key=key)
        if way_ids_of_edge(graph, edge) & via:
            nodes |= {source, target}
    return nodes


def build_review_cards(inputs: ReviewInputs) -> tuple[list[StreetCard], list[JunctionCard]]:
    """Return the street cards and junction cards of a graph."""
    graph = inputs.graph
    regular = [s for s in inputs.streets if s.category is StreetCategory.REGULAR]
    junction_by_id = {j.junction_id: j for j in inputs.junctions}
    regular_names = {s.street_id: s.display_name for s in regular}
    junction_ids_of_street: dict[str, list[str]] = defaultdict(list)
    for junction in inputs.junctions:
        for street_id in junction.street_ids:
            junction_ids_of_street[street_id].append(junction.junction_id)
    place_names = {name for j in inputs.junctions for name in j.place_names}
    junction_of_node = {n: j.junction_id for j in inputs.junctions for n in j.node_ids}
    street_cards = [
        street_card(
            inputs, street, junction_ids_of_street, junction_by_id, regular_names, junction_of_node
        )
        for street in sorted(regular, key=lambda s: s.display_name)
        if street.display_name not in place_names
    ]

    restriction_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for restriction in inputs.restrictions:
        if not restriction.applies_to_cars:
            continue
        touched = {
            junction_of_node[n]
            for n in restriction_nodes(graph, restriction)
            if n in junction_of_node
        }
        status = map_restriction(graph, restriction).status
        if status in (MappingStatus.OUTSIDE_GRAPH, MappingStatus.ALREADY_IMPOSSIBLE):
            continue
        mapped = status is MappingStatus.MAPPED
        for junction_id in touched:
            if not mapped:
                restriction_counts[junction_id]["unmapped"] += 1
            elif restriction.is_unconditional:
                restriction_counts[junction_id]["unconditional"] += 1
            else:
                restriction_counts[junction_id]["conditional"] += 1
    junctions_of_place: Counter[str] = Counter(
        name for j in inputs.junctions for name in j.place_names
    )
    nodes_of_street: dict[str, set[int]] = defaultdict(set)
    for street in regular:
        for edge in street.edges:
            nodes_of_street[street.street_id] |= {edge.source_node_id, edge.target_node_id}

    junction_cards = []
    for junction in inputs.junctions:
        street_ids = [s for s in junction.street_ids if s in regular_names]
        if len(street_ids) < 2 and not junction.place_names:
            continue
        flags: list[ReviewFlag] = []
        if junction.spread_m > WIDE_JUNCTION_M:
            flags.append(ReviewFlag(code="J-WIDE", message=f"spread {junction.spread_m:.0f} m"))
        if len(junction.node_ids) >= MANY_NODES:
            flags.append(ReviewFlag(code="J-MANY-NODES", message=f"{len(junction.node_ids)} nodes"))
        for name in junction.place_names:
            if junctions_of_place[name] > 1:
                flags.append(
                    ReviewFlag(code="J-PLACE-SPLIT", message=f"{name} forms several junctions")
                )
        if junction.kind is JunctionKind.WITH_CONNECTORS and junction.spread_m > CONNECTOR_SPREAD_M:
            flags.append(
                ReviewFlag(
                    code="J-CONNECTORS",
                    message=f"unnamed connectors joined over {junction.spread_m:.0f} m",
                )
            )
        new_pairs = sorted(
            f"{regular_names[a]} / {regular_names[b]}"
            for i, a in enumerate(street_ids)
            for b in street_ids[i + 1 :]
            if not (nodes_of_street[a] & nodes_of_street[b])
        )
        if new_pairs:
            flags.append(
                ReviewFlag(
                    code="J-MEET-WITHOUT-SHARED-NODE",
                    message="meet here without a shared node: " + "; ".join(new_pairs),
                )
            )
        counts = restriction_counts.get(junction.junction_id, Counter())
        if counts["unmapped"]:
            flags.append(
                ReviewFlag(
                    code="J-UNMAPPED-RESTRICTION",
                    message=f"{counts['unmapped']} car restriction(s) not mapped onto edges",
                )
            )
        junction_cards.append(
            JunctionCard(
                junction_id=junction.junction_id,
                kind=junction.kind.value,
                node_count=len(junction.node_ids),
                spread_m=round(junction.spread_m, 1),
                place_names=junction.place_names,
                regular_street_names=sorted(regular_names[s] for s in street_ids),
                unconditional_car_restrictions=counts["unconditional"],
                conditional_car_restrictions=counts["conditional"],
                unmapped_restrictions=counts["unmapped"],
                latitude=junction.latitude,
                longitude=junction.longitude,
                flags=flags,
            )
        )
    return street_cards, junction_cards
