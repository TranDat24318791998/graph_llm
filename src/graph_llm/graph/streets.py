"""Build the street layer: group the named edges of the road graph into streets.

OSM often stores one street as several disconnected pieces: a divided road is
drawn as two one-way ways, and a street can be interrupted at a junction whose
segment carries another name or no name. The rules, applied per name key, are:

1. Split the edges of the name into connected pieces, ignoring edge direction.
2. Join two pieces if a path of at most ``merge_gap_m`` metres connects them,
   ignoring edge direction.
3. If more than one group remains, a reviewed override must decide: ``merge``
   makes all pieces one street, ``separate`` keeps the groups as different streets
   that share a name. A name without an override is an error, so every such case
   is decided explicitly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import networkx as nx

from graph_llm.graph.coverage import as_list, has_value
from graph_llm.graph.types import RoadGraph
from graph_llm.names.normalize import ascii_key, clean_name, is_alley_name, name_key, slugify
from graph_llm.schemas.question import EdgeRef
from graph_llm.schemas.street import (
    GraphProcessingConfig,
    NameResolution,
    OverrideDecision,
    Street,
    StreetCategory,
)


@dataclass(frozen=True)
class EdgeSpelling:
    """One OSM spelling of a name on one edge."""

    edge: EdgeRef
    spelling: str
    length_m: float


@dataclass(frozen=True)
class NamedEdges:
    """The edges that carry one name key, and the OSM spellings on each of them."""

    edges: list[EdgeRef]
    spellings: list[EdgeSpelling]


@dataclass(frozen=True)
class StreetLayer:
    """The streets of a road graph and how multi-piece names were resolved."""

    streets: list[Street]
    name_resolutions: list[NameResolution]
    unnamed_edge_count: int


def collect_named_edges(graph: RoadGraph) -> tuple[dict[str, NamedEdges], int]:
    """Group edges by name key. An edge with several names belongs to each of them.

    Returns:
        The edges per name key, and the number of edges without a name.
    """
    edges_by_key: dict[str, list[EdgeRef]] = defaultdict(list)
    spellings_by_key: dict[str, list[EdgeSpelling]] = defaultdict(list)
    unnamed_edge_count = 0

    for source, target, key, attributes in graph.edges(keys=True, data=True):
        raw_names = attributes.get("name")
        if not has_value(raw_names):
            unnamed_edge_count += 1
            continue
        edge = EdgeRef(source_node_id=source, target_node_id=target, key=key)
        keys_of_edge: set[str] = set()
        for raw_name in as_list(raw_names):
            spelling = clean_name(str(raw_name))
            key_of_name = name_key(spelling)
            spellings_by_key[key_of_name].append(
                EdgeSpelling(edge=edge, spelling=spelling, length_m=float(attributes["length"]))
            )
            keys_of_edge.add(key_of_name)
        for key_of_name in keys_of_edge:
            edges_by_key[key_of_name].append(edge)

    named_edges = {
        key_of_name: NamedEdges(edges=edges, spellings=spellings_by_key[key_of_name])
        for key_of_name, edges in edges_by_key.items()
    }
    return named_edges, unnamed_edge_count


def split_into_pieces(edges: list[EdgeRef]) -> list[set[int]]:
    """Return the node sets of the connected pieces formed by the edges, ignoring direction."""
    piece_graph: nx.Graph[int] = nx.Graph()
    piece_graph.add_edges_from((edge.source_node_id, edge.target_node_id) for edge in edges)
    return [set(piece) for piece in nx.connected_components(piece_graph)]


def group_pieces_within_gap(
    pieces: list[set[int]], undirected_graph: nx.MultiGraph[int], max_gap_m: float
) -> list[list[int]]:
    """Join pieces that a path of at most ``max_gap_m`` metres connects.

    Returns:
        Groups of piece indices. Joining is transitive.
    """
    piece_of_node = {node: index for index, piece in enumerate(pieces) for node in piece}
    parent = list(range(len(pieces)))

    def find_root(index: int) -> int:
        while parent[index] != index:
            index = parent[index]
        return index

    for index, piece in enumerate(pieces):
        reachable = nx.multi_source_dijkstra_path_length(
            undirected_graph, piece, cutoff=max_gap_m, weight="length"
        )
        for node in reachable:
            other_index = piece_of_node.get(node)
            if other_index is not None:
                parent[find_root(other_index)] = find_root(index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(pieces)):
        groups[find_root(index)].append(index)
    return list(groups.values())


def mean_latitude(graph: RoadGraph, nodes: set[int]) -> float:
    """Return the mean latitude of a set of nodes."""
    return sum(float(graph.nodes[node]["y"]) for node in nodes) / len(nodes)


def choose_display_name(spelling_lengths_m: Mapping[str, float]) -> str:
    """Return the spelling that covers the most length; ties go to the first in sort order."""
    return min(spelling_lengths_m, key=lambda spelling: (-spelling_lengths_m[spelling], spelling))


def resolve_groups(
    key_of_name: str,
    pieces: list[set[int]],
    automatic_groups: list[list[int]],
    overrides: dict[str, OverrideDecision],
) -> tuple[list[set[int]], OverrideDecision | None]:
    """Apply the override of a name to the groups left by the automatic rule.

    Returns:
        The node set of each final street, and the override that was applied.

    Raises:
        ValueError: If several groups remain and no override decides the name.
    """
    group_nodes = [set().union(*(pieces[index] for index in group)) for group in automatic_groups]
    if len(group_nodes) == 1:
        return group_nodes, None

    decision = overrides.get(key_of_name)
    if decision is None:
        raise ValueError(f"Name '{key_of_name}' has {len(group_nodes)} separate groups.")
    if decision is OverrideDecision.MERGE:
        return [set().union(*group_nodes)], decision
    return group_nodes, decision


def build_street_layer(graph: RoadGraph, config: GraphProcessingConfig) -> StreetLayer:
    """Group the named edges of the graph into streets.

    Raises:
        ValueError: If a name needs an override and has none, if an override, an
            internal road name or an alias refers to a name that is not in the
            graph, or if an override is not needed.
    """
    named_edges, unnamed_edge_count = collect_named_edges(graph)
    overrides = {name_key(item.name): item.decision for item in config.street_overrides}
    internal_keys = {name_key(name) for name in config.internal_road_names}
    aliases_by_key = {
        name_key(item.name): [clean_name(alias) for alias in item.aliases]
        for item in config.name_aliases
    }
    check_names_exist(named_edges, overrides.keys(), "street override")
    check_names_exist(named_edges, internal_keys, "internal road name")
    check_names_exist(named_edges, aliases_by_key.keys(), "alias entry")
    check_aliases_are_unambiguous(named_edges, aliases_by_key)

    undirected_graph = graph.to_undirected(as_view=True)
    streets: list[Street] = []
    resolutions: list[NameResolution] = []
    unresolved: list[str] = []
    used_overrides: set[str] = set()

    for key_of_name in sorted(named_edges):
        named = named_edges[key_of_name]
        pieces = split_into_pieces(named.edges)
        automatic_groups = (
            group_pieces_within_gap(pieces, undirected_graph, config.merge_gap_m)
            if len(pieces) > 1
            else [[0]]
        )
        try:
            street_nodes, applied = resolve_groups(key_of_name, pieces, automatic_groups, overrides)
        except ValueError as error:
            unresolved.append(str(error))
            continue
        if applied is not None:
            used_overrides.add(key_of_name)

        street_nodes.sort(key=lambda nodes: -mean_latitude(graph, nodes))
        new_streets = make_streets(
            key_of_name, named, street_nodes, internal_keys, aliases_by_key.get(key_of_name, [])
        )
        streets.extend(new_streets)
        if len(pieces) > 1:
            resolutions.append(
                NameResolution(
                    name_key=key_of_name,
                    piece_count=len(pieces),
                    groups_after_automatic_rule=len(automatic_groups),
                    override=applied,
                    street_ids=[street.street_id for street in new_streets],
                )
            )

    if unresolved:
        raise ValueError("Names that need a street override:\n" + "\n".join(unresolved))
    unused_overrides = sorted(set(overrides) - used_overrides)
    if unused_overrides:
        raise ValueError(
            f"Street overrides that the automatic rule makes unnecessary: {unused_overrides}"
        )
    check_street_ids_are_unique(streets)
    return StreetLayer(
        streets=streets, name_resolutions=resolutions, unnamed_edge_count=unnamed_edge_count
    )


def make_streets(
    key_of_name: str,
    named: NamedEdges,
    street_nodes: list[set[int]],
    internal_keys: set[str],
    aliases: list[str],
) -> list[Street]:
    """Create one street per node set. Several streets of one name get numbered ids."""
    if is_alley_name(key_of_name):
        category = StreetCategory.ALLEY
    elif key_of_name in internal_keys:
        category = StreetCategory.INTERNAL
    else:
        category = StreetCategory.REGULAR

    all_spelling_lengths_m: dict[str, float] = defaultdict(float)
    for item in named.spellings:
        all_spelling_lengths_m[item.spelling] += item.length_m
    base_id = slugify(choose_display_name(all_spelling_lengths_m))
    is_shared = len(street_nodes) > 1
    streets = []
    for number, nodes in enumerate(street_nodes, start=1):
        edges = [edge for edge in named.edges if edge.source_node_id in nodes]
        spelling_lengths_m: dict[str, float] = defaultdict(float)
        for item in named.spellings:
            if item.edge.source_node_id in nodes:
                spelling_lengths_m[item.spelling] += item.length_m
        display_name = choose_display_name(spelling_lengths_m)
        streets.append(
            Street(
                street_id=f"{base_id}-{number}" if is_shared else base_id,
                display_name=display_name,
                name_key=key_of_name,
                ascii_key=ascii_key(display_name),
                category=category,
                osm_names=sorted(spelling_lengths_m),
                aliases=aliases,
                shares_name_with_other_street=is_shared,
                edges=edges,
            )
        )
    return streets


def check_names_exist(
    named_edges: dict[str, NamedEdges], keys: Iterable[str], description: str
) -> None:
    """Raise if a configured name does not occur in the graph."""
    missing = sorted(key for key in keys if key not in named_edges)
    if missing:
        raise ValueError(f"Every {description} must name a street in the graph; unknown: {missing}")


def check_aliases_are_unambiguous(
    named_edges: dict[str, NamedEdges], aliases_by_key: dict[str, list[str]]
) -> None:
    """Raise if an alias equals another street name or is given for two names."""
    owner_of_alias: dict[str, str] = {}
    for key_of_name, aliases in aliases_by_key.items():
        for alias in aliases:
            alias_key = name_key(alias)
            if alias_key in named_edges and alias_key != key_of_name:
                raise ValueError(f"Alias '{alias}' of '{key_of_name}' is another street's name.")
            previous_owner = owner_of_alias.setdefault(alias_key, key_of_name)
            if previous_owner != key_of_name:
                raise ValueError(
                    f"Alias '{alias}' is given for '{previous_owner}' and '{key_of_name}'."
                )


def check_street_ids_are_unique(streets: list[Street]) -> None:
    """Raise if two different names produce the same street id."""
    counts = Counter(street.street_id for street in streets)
    duplicates = sorted(street_id for street_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Street ids are not unique: {duplicates}")
