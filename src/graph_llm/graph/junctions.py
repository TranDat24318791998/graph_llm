"""Group road-layer nodes into junctions: the real intersections people talk about.

One real intersection is often several OSM nodes: a roundabout ring, the two
crossings of a divided road, slip lanes, or a named square. A junction is a
connected group of nodes joined by junction-internal edges:

- edges of a roundabout ring (``junction=roundabout`` or ``circular``);
- edges of a place: a named square or roundabout such as "Công trường Quốc Tế";
- unnamed connecting edges (``*_link``, or unnamed edges of a main road class)
  up to ``link_max_m``;
- median crossings: an edge up to ``median_crossing_max_m`` whose two ends lie
  on the two carriageways of one divided street;
- tiny edges up to ``tiny_edge_max_m`` between two intersection nodes.

Nodes are never grouped because they are close; only along such edges, so a
bridge passing over a street does not become an intersection. Every other node
where three or more road edges meet, or where two named streets meet, is a
junction on its own. Two streets meet when they reach the same junction.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx

from graph_llm.graph.carriageways import (
    EdgeDirection,
    StreetEdgeLabels,
    edge_polyline,
    midpoint_and_heading,
)
from graph_llm.graph.types import RoadGraph
from graph_llm.names.normalize import name_key
from graph_llm.schemas.junction import Junction, JunctionKind, JunctionSettings
from graph_llm.schemas.question import EdgeRef
from graph_llm.schemas.street import Street, StreetCategory

RING_JUNCTION_VALUES = {"roundabout", "circular"}
MAIN_ROAD_CLASSES = {"trunk", "primary", "secondary", "tertiary", "unclassified"}
METRES_PER_DEGREE = 111_320.0


def first_value(value: object) -> str | None:
    """Return a single attribute value; for a list (merged OSM ways) the first item."""
    if isinstance(value, list):
        return str(value[0]) if value else None
    return None if value is None else str(value)


def is_place_name(name: str, settings: JunctionSettings) -> bool:
    """Return True if a street name names a square or roundabout (a place)."""
    key = name_key(name)
    return any(key.startswith(name_key(prefix)) for prefix in settings.place_name_prefixes)


def place_street_ids(streets: Sequence[Street], settings: JunctionSettings) -> set[str]:
    """Return the streets that are places (named squares and roundabouts), not streets.

    Names listed in ``street_like_places`` keep their place-like name but are streets.
    """
    street_like = {name_key(name) for name in settings.street_like_places}
    return {
        street.street_id
        for street in streets
        if is_place_name(street.display_name, settings)
        and name_key(street.display_name) not in street_like
    }


def intersection_nodes(graph: RoadGraph, street_ids_at_node: Mapping[int, set[str]]) -> set[int]:
    """Return nodes where three or more road edges meet, or two named streets meet."""
    undirected_degree = {
        node: len(set(graph.successors(node)) | set(graph.predecessors(node)))
        for node in graph.nodes
    }
    return {
        node
        for node in graph.nodes
        if undirected_degree[node] >= 3 or len(street_ids_at_node.get(node, set())) >= 2
    }


def divided_heading_at(
    graph: RoadGraph,
    node: int,
    street_id: str,
    street_ids_of_edge: Mapping[EdgeRef, list[str]],
    labels: StreetEdgeLabels,
) -> float | None:
    """Return the heading of a divided carriageway of a street through a node, if any."""
    for source, target, key in list(graph.in_edges(node, keys=True)) + list(
        graph.out_edges(node, keys=True)
    ):
        edge = EdgeRef(source_node_id=source, target_node_id=target, key=key)
        if (
            street_id in street_ids_of_edge.get(edge, [])
            and labels.get(street_id, {}).get(edge) is EdgeDirection.DIVIDED
        ):
            return midpoint_and_heading(edge_polyline(graph, edge))[1]
    return None


def is_median_crossing(
    graph: RoadGraph,
    edge: EdgeRef,
    street_ids_at_node: Mapping[int, set[str]],
    street_ids_of_edge: Mapping[EdgeRef, list[str]],
    labels: StreetEdgeLabels,
) -> bool:
    """Return True if an edge connects the two carriageways of a divided street."""
    own = set(street_ids_of_edge.get(edge, []))
    shared = (
        street_ids_at_node.get(edge.source_node_id, set())
        & street_ids_at_node.get(edge.target_node_id, set())
    ) - own
    for street_id in shared:
        first = divided_heading_at(
            graph, edge.source_node_id, street_id, street_ids_of_edge, labels
        )
        second = divided_heading_at(
            graph, edge.target_node_id, street_id, street_ids_of_edge, labels
        )
        if first is not None and second is not None and math.cos(first - second) <= -0.8:
            return True
    return False


@dataclass(frozen=True)
class InternalEdge:
    """A junction-internal edge and why it is internal."""

    edge: EdgeRef
    reason: str


def internal_edges(
    graph: RoadGraph,
    streets: Sequence[Street],
    labels: StreetEdgeLabels,
    settings: JunctionSettings,
) -> list[InternalEdge]:
    """Return every junction-internal edge with its reason."""
    street_ids_of_edge: dict[EdgeRef, list[str]] = defaultdict(list)
    street_ids_at_node: dict[int, set[str]] = defaultdict(set)
    for street in streets:
        for edge in street.edges:
            street_ids_of_edge[edge].append(street.street_id)
            street_ids_at_node[edge.source_node_id].add(street.street_id)
            street_ids_at_node[edge.target_node_id].add(street.street_id)
    places = place_street_ids(streets, settings)
    intersections = intersection_nodes(graph, street_ids_at_node)
    result = []
    for source, target, key, data in graph.edges(keys=True, data=True):
        edge = EdgeRef(source_node_id=source, target_node_id=target, key=key)
        owners = street_ids_of_edge.get(edge, [])
        length_m = float(data["length"])
        highway = first_value(data.get("highway")) or ""
        reason = None
        if first_value(data.get("junction")) in RING_JUNCTION_VALUES:
            reason = "roundabout ring"
        elif any(owner in places for owner in owners) and (
            all(owner in places for owner in owners) or length_m <= settings.link_max_m
        ):
            reason = "place"
        elif (
            not owners
            and length_m <= settings.link_max_m
            and (highway.endswith("_link") or highway in MAIN_ROAD_CLASSES)
        ):
            reason = "unnamed connector"
        elif (
            length_m <= settings.median_crossing_max_m
            and source in intersections
            and target in intersections
            and is_median_crossing(graph, edge, street_ids_at_node, street_ids_of_edge, labels)
        ):
            reason = "median crossing"
        elif (
            length_m <= settings.tiny_edge_max_m
            and source in intersections
            and target in intersections
        ):
            reason = "tiny edge"
        if reason is not None:
            result.append(InternalEdge(edge, reason))
    return result


def centroid(graph: RoadGraph, nodes: set[int]) -> tuple[float, float]:
    """Return the mean (latitude, longitude) of nodes."""
    latitude = sum(float(graph.nodes[n]["y"]) for n in nodes) / len(nodes)
    longitude = sum(float(graph.nodes[n]["x"]) for n in nodes) / len(nodes)
    return latitude, longitude


def spread_m(graph: RoadGraph, nodes: set[int]) -> float:
    """Return the largest distance in metres between two nodes of a group."""
    points = [(float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])) for n in nodes]
    if len(points) < 2:
        return 0.0
    scale = math.cos(math.radians(points[0][1])) * METRES_PER_DEGREE
    return max(
        math.hypot((a[0] - b[0]) * scale, (a[1] - b[1]) * METRES_PER_DEGREE)
        for i, a in enumerate(points)
        for b in points[i + 1 :]
    )


def merge_groups_of_one_place(
    groups: list[set[int]], street_ids_at_node: Mapping[int, set[str]], places: set[str]
) -> list[set[int]]:
    """Merge node groups that belong to the same place (a square drawn in several pieces)."""
    merged: nx.Graph[int] = nx.Graph()
    for index in range(len(groups)):
        merged.add_node(index)
    first_group_of_place: dict[str, int] = {}
    for index, nodes in enumerate(groups):
        for node in nodes:
            for place in street_ids_at_node.get(node, set()) & places:
                if place in first_group_of_place:
                    merged.add_edge(first_group_of_place[place], index)
                else:
                    first_group_of_place[place] = index
    return [
        set().union(*(groups[i] for i in component))
        for component in nx.connected_components(merged)
    ]


def merge_split_crossings(
    graph: RoadGraph,
    groups: list[set[int]],
    street_ids_at_node: Mapping[int, set[str]],
    regular_ids: set[str],
    settings: JunctionSettings,
) -> tuple[list[set[int]], set[int]]:
    """Merge a junction into a neighbouring one when both are the same crossing.

    Group A joins group B when A's regular streets (at least two) are all among B's,
    and an edge of at most ``split_crossing_link_max_m`` joins a node of A to a node of
    B. This catches corners of one crossing that slip lanes or medians split into
    several node groups; it never merges by distance alone.

    Returns:
        The merged groups and the nodes of groups that were merged.
    """
    group_of_node = {node: index for index, nodes in enumerate(groups) for node in nodes}
    streets_of_group = [
        set().union(*(street_ids_at_node.get(n, set()) for n in nodes)) & regular_ids
        for nodes in groups
    ]
    links: nx.Graph[int] = nx.Graph()
    links.add_nodes_from(range(len(groups)))
    for source, target, data in graph.edges(data=True):
        if float(data["length"]) > settings.split_crossing_link_max_m:
            continue
        first, second = group_of_node.get(source), group_of_node.get(target)
        if first is None or second is None or first == second:
            continue
        smaller, larger = sorted((first, second), key=lambda i: len(streets_of_group[i]))
        if (
            len(streets_of_group[smaller]) >= 2
            and streets_of_group[smaller] <= streets_of_group[larger]
        ):
            links.add_edge(smaller, larger)
    merged_nodes: set[int] = set()
    result = []
    for component in nx.connected_components(links):
        nodes = set().union(*(groups[i] for i in component))
        if len(component) > 1:
            merged_nodes |= nodes
        result.append(nodes)
    return result, merged_nodes


def build_junctions(
    graph: RoadGraph,
    streets: Sequence[Street],
    labels: StreetEdgeLabels,
    settings: JunctionSettings,
) -> list[Junction]:
    """Return the junctions of a road graph, ordered by their smallest node id."""
    street_ids_at_node: dict[int, set[str]] = defaultdict(set)
    for street in streets:
        for edge in street.edges:
            street_ids_at_node[edge.source_node_id].add(street.street_id)
            street_ids_at_node[edge.target_node_id].add(street.street_id)
    places = place_street_ids(streets, settings)
    display_names = {street.street_id: street.display_name for street in streets}
    internal = internal_edges(graph, streets, labels, settings)
    grouping: nx.Graph[int] = nx.Graph()
    reasons_by_node: dict[int, set[str]] = defaultdict(set)
    for item in internal:
        grouping.add_edge(item.edge.source_node_id, item.edge.target_node_id)
        reasons_by_node[item.edge.source_node_id].add(item.reason)
        reasons_by_node[item.edge.target_node_id].add(item.reason)
    groups = [set(component) for component in nx.connected_components(grouping)]
    grouped = set().union(*groups) if groups else set()
    groups += [{node} for node in sorted(intersection_nodes(graph, street_ids_at_node) - grouped)]
    groups = merge_groups_of_one_place(groups, street_ids_at_node, places)
    regular_ids = {
        street.street_id
        for street in streets
        if street.category is StreetCategory.REGULAR and street.street_id not in places
    }
    groups, merged_nodes = merge_split_crossings(
        graph, groups, street_ids_at_node, regular_ids, settings
    )
    for node in merged_nodes:
        reasons_by_node[node].add("split crossing")

    junctions = []
    for nodes in sorted(groups, key=min):
        reasons = set().union(*(reasons_by_node.get(n, set()) for n in nodes))
        streets_here = set().union(*(street_ids_at_node.get(n, set()) for n in nodes))
        place_ids = sorted(streets_here & places)
        if "roundabout ring" in reasons or "place" in reasons:
            kind = JunctionKind.ROUNDABOUT_OR_PLACE
        elif "median crossing" in reasons:
            kind = JunctionKind.DIVIDED_CROSSING
        elif "unnamed connector" in reasons or "split crossing" in reasons:
            kind = JunctionKind.WITH_CONNECTORS
        elif len(nodes) > 1:
            kind = JunctionKind.COMPACT
        else:
            kind = JunctionKind.SINGLE_NODE
        latitude, longitude = centroid(graph, nodes)
        junctions.append(
            Junction(
                junction_id=f"j{min(nodes)}",
                node_ids=sorted(nodes),
                latitude=latitude,
                longitude=longitude,
                spread_m=spread_m(graph, nodes),
                kind=kind,
                internal_edge_reasons=sorted(reasons),
                place_names=[display_names[p] for p in place_ids],
                street_ids=sorted(streets_here - places),
            )
        )
    return junctions
