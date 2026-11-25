"""Simplify an unsimplified OSM drive graph without merging different roads.

OSMnx's default simplification removes every node that is not an intersection or
a dead end and merges the edges around it, even when they carry different names
or road classes; the merged edge then stores lists of values and cannot be split
again. Graph v2 starts from an unsimplified snapshot and keeps a node wherever:

- the incident edges differ in any attribute of ``EDGE_ATTRIBUTES_THAT_SPLIT``
  (name, road class, direction, roundabout, bridge, tunnel, lanes, speed, access);
- a turn restriction acts: its via node, or any node of its via way(s), so that
  every restriction maps onto whole edges.

The result has the same roads and lengths as the input; only interstitial nodes
inside one uniform stretch of road are removed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import osmnx

from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.restriction import TurnRestriction

EDGE_ATTRIBUTES_THAT_SPLIT = [
    "name",
    "highway",
    "oneway",
    "junction",
    "bridge",
    "tunnel",
    "lanes",
    "maxspeed",
    "access",
]
KEEP_NODE_ATTRIBUTE = "keep_for_restriction"


def edge_way_ids(data: dict[str, object]) -> set[int]:
    """Return the OSM way id(s) of an edge."""
    raw = data.get("osmid")
    values: Iterable[object] = raw if isinstance(raw, list) else [raw]
    return {int(str(value)) for value in values if value is not None}


def restriction_node_ids(graph: RoadGraph, restrictions: Sequence[TurnRestriction]) -> set[int]:
    """Return the nodes that must survive simplification because a restriction acts there."""
    via_ways = {way for r in restrictions for way in r.via_way_ids}
    nodes = {node for r in restrictions for node in r.via_node_ids if node in graph}
    for source, target, data in graph.edges(data=True):
        if edge_way_ids(data) & via_ways:
            nodes |= {source, target}
    return nodes


def simplify_keeping_roads_apart(
    graph: RoadGraph, restrictions: Sequence[TurnRestriction]
) -> RoadGraph:
    """Return a simplified copy of an unsimplified graph (see the module docstring).

    Raises:
        ValueError: If the graph is already simplified.
    """
    if graph.graph.get("simplified"):
        raise ValueError("The graph is already simplified; start from an unsimplified snapshot.")
    working = graph.copy()
    for node in restriction_node_ids(working, restrictions):
        working.nodes[node][KEEP_NODE_ATTRIBUTE] = True
    simplified: RoadGraph = osmnx.simplify_graph(
        working,
        node_attrs_include=[KEEP_NODE_ATTRIBUTE],
        edge_attrs_differ=EDGE_ATTRIBUTES_THAT_SPLIT,
    )
    for node in simplified.nodes:
        simplified.nodes[node].pop(KEEP_NODE_ATTRIBUTE, None)
    return simplified
