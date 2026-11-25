"""Classify every street edge by how traffic uses it: two-way, divided or one-way.

OSM stores a divided road (two carriageways with a median) as two one-way ways
in opposite directions. An edge is labelled:

- ``TWO_WAY`` if the reverse edge (same nodes) belongs to the same street;
- ``DIVIDED`` if it is one-way and an opposite one-way edge of the same street
  runs alongside it: the distance from its midpoint to that edge's polyline is at
  most ``max_separation_m`` and their directions there are nearly opposite. Where
  the carriageways split or rejoin, the two edges share a node; such a pair counts
  only if the midpoint is at least 4 m from the other edge (a median and a lane),
  because closer opposite edges that meet at a node form a hairpin;
- ``ONE_WAY`` otherwise.

Pairing works edge by edge, not segment by segment, because side streets often
join only one carriageway, so the two carriageways are split at different nodes.
Labels belong to a (street, edge) pair: an edge that carries two street names is
classified separately within each street.

From the labels a street gets a direction status (``StreetDirection``) and a
centre-line length: a two-way pair counts once, the two carriageways of a
divided road count as their mean, a one-way edge counts fully.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum

from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.question import EdgeRef
from graph_llm.schemas.street import Street

METRES_PER_DEGREE = 111_320.0
DEFAULT_MAX_SEPARATION_M = 40.0
OPPOSITE_COSINE = -0.8
MIN_SEPARATION_AT_SHARED_NODE_M = 4.0


class EdgeDirection(StrEnum):
    """How traffic uses one edge of a street."""

    TWO_WAY = "two_way"
    DIVIDED = "divided"
    ONE_WAY = "one_way"


class StreetDirection(StrEnum):
    """Direction status of a whole street."""

    TWO_WAY = "two_way"
    TWO_WAY_DIVIDED = "two_way_divided"
    ONE_WAY = "one_way"
    MIXED = "mixed"


Point = tuple[float, float]


def edge_polyline(graph: RoadGraph, edge: EdgeRef) -> list[Point]:
    """Return the (longitude, latitude) points of an edge, from its geometry if present."""
    data = graph.edges[edge.source_node_id, edge.target_node_id, edge.key]
    geometry = data.get("geometry")
    if geometry is not None and hasattr(geometry, "coords"):
        return [(float(x), float(y)) for x, y in geometry.coords]
    source = graph.nodes[edge.source_node_id]
    target = graph.nodes[edge.target_node_id]
    return [(float(source["x"]), float(source["y"])), (float(target["x"]), float(target["y"]))]


def to_metres(point: Point, origin: Point) -> tuple[float, float]:
    """Project a point to local metres around an origin (equirectangular)."""
    scale = math.cos(math.radians(origin[1])) * METRES_PER_DEGREE
    return (point[0] - origin[0]) * scale, (point[1] - origin[1]) * METRES_PER_DEGREE


def midpoint_and_heading(points: Sequence[Point]) -> tuple[Point, float]:
    """Return the point halfway along a polyline and the heading of the piece it lies on."""
    origin = points[0]
    local = [to_metres(p, origin) for p in points]
    pieces = [math.dist(a, b) for a, b in zip(local, local[1:], strict=False)]
    target = sum(pieces) / 2
    for index, piece in enumerate(pieces):
        if target <= piece or index == len(pieces) - 1:
            a, b = points[index], points[index + 1]
            t = 0.0 if piece == 0 else min(1.0, target / piece)
            la, lb = local[index], local[index + 1]
            heading = math.atan2(lb[1] - la[1], lb[0] - la[0])
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), heading
        target -= piece
    raise ValueError("A polyline needs at least two points.")


def distance_and_heading_to(point: Point, polyline: Sequence[Point]) -> tuple[float, float]:
    """Return the distance in metres from a point to a polyline, and the polyline heading there."""
    best = (math.inf, 0.0)
    p = (0.0, 0.0)
    for a, b in zip(polyline, polyline[1:], strict=False):
        la, lb = to_metres(a, point), to_metres(b, point)
        dx, dy = lb[0] - la[0], lb[1] - la[1]
        length_squared = dx * dx + dy * dy
        t = (
            0.0
            if length_squared == 0
            else ((p[0] - la[0]) * dx + (p[1] - la[1]) * dy) / length_squared
        )
        t = max(0.0, min(1.0, t))
        closest = (la[0] + t * dx, la[1] + t * dy)
        distance = math.dist(p, closest)
        if distance < best[0]:
            best = (distance, math.atan2(dy, dx))
    return best


def classify_street_edges(
    graph: RoadGraph,
    edges: Sequence[EdgeRef],
    max_separation_m: float = DEFAULT_MAX_SEPARATION_M,
) -> dict[EdgeRef, EdgeDirection]:
    """Label every edge of one street as two-way, divided or one-way."""
    node_pairs = {(edge.source_node_id, edge.target_node_id) for edge in edges}
    one_way_edges = []
    labels: dict[EdgeRef, EdgeDirection] = {}
    for edge in edges:
        if (edge.target_node_id, edge.source_node_id) in node_pairs:
            labels[edge] = EdgeDirection.TWO_WAY
        else:
            one_way_edges.append(edge)
    polylines = {edge: edge_polyline(graph, edge) for edge in one_way_edges}
    for edge in one_way_edges:
        middle, heading = midpoint_and_heading(polylines[edge])
        paired = False
        for other in one_way_edges:
            if other == edge:
                continue
            distance, other_heading = distance_and_heading_to(middle, polylines[other])
            shares_node = bool(
                {edge.source_node_id, edge.target_node_id}
                & {other.source_node_id, other.target_node_id}
            )
            if shares_node and distance < MIN_SEPARATION_AT_SHARED_NODE_M:
                # Opposite edges that meet at a node and lie this close form a hairpin
                # or a drawing artefact, not two carriageways with a median.
                continue
            if (
                distance <= max_separation_m
                and math.cos(other_heading - heading) <= OPPOSITE_COSINE
            ):
                paired = True
                break
        labels[edge] = EdgeDirection.DIVIDED if paired else EdgeDirection.ONE_WAY
    return labels


def is_consistent_one_way_path(edges: Sequence[EdgeRef]) -> bool:
    """Return True if one-way edges form one simple path oriented in one direction."""
    out_degree: dict[int, int] = {}
    in_degree: dict[int, int] = {}
    for edge in edges:
        out_degree[edge.source_node_id] = out_degree.get(edge.source_node_id, 0) + 1
        in_degree[edge.target_node_id] = in_degree.get(edge.target_node_id, 0) + 1
    nodes = set(out_degree) | set(in_degree)
    starts = [n for n in nodes if in_degree.get(n, 0) == 0]
    return (
        all(count <= 1 for count in out_degree.values())
        and all(count <= 1 for count in in_degree.values())
        and len(starts) == 1
        and len(edges) == len(nodes) - 1
    )


def street_direction(labels: Mapping[EdgeRef, EdgeDirection]) -> StreetDirection:
    """Return the direction status of a street from its edge labels."""
    kinds = set(labels.values())
    if kinds == {EdgeDirection.TWO_WAY}:
        return StreetDirection.TWO_WAY
    if kinds and kinds <= {EdgeDirection.TWO_WAY, EdgeDirection.DIVIDED}:
        return StreetDirection.TWO_WAY_DIVIDED
    if kinds == {EdgeDirection.ONE_WAY} and is_consistent_one_way_path(list(labels)):
        return StreetDirection.ONE_WAY
    return StreetDirection.MIXED


def centre_line_length_m(
    labels: Mapping[EdgeRef, EdgeDirection], lengths_m: Mapping[EdgeRef, float]
) -> float:
    """Return a street's length along its centre line (each stretch of road counted once)."""
    total = 0.0
    for edge, label in labels.items():
        if label is EdgeDirection.ONE_WAY:
            total += lengths_m[edge]
        else:
            total += lengths_m[edge] / 2
    return total


StreetEdgeLabels = Mapping[str, Mapping[EdgeRef, EdgeDirection]]


def classify_all_streets(
    graph: RoadGraph,
    streets: Sequence[Street],
    max_separation_m: float = DEFAULT_MAX_SEPARATION_M,
) -> dict[str, dict[EdgeRef, EdgeDirection]]:
    """Return the edge labels of every street, keyed by street id."""
    return {
        street.street_id: classify_street_edges(graph, street.edges, max_separation_m)
        for street in streets
    }
