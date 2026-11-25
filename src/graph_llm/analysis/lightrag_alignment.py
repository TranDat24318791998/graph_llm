"""Compare the knowledge graph LightRAG built (M2) with the OSM street graph.

LightRAG entities are matched to streets by name (``StreetLinker.resolve``). A
relation between two entities that both match streets is correct if some pair of
those streets meets in the OSM graph. Recall is measured over the pairs of
regular streets that meet. Direction is checked by looking for one-way wording
in the descriptions, because the LightRAG graph itself is undirected.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from graph_llm.graph.carriageways import StreetDirection
from graph_llm.questions.graph_view import GraphView
from graph_llm.representations.linker import StreetLinker
from graph_llm.schemas.base import StrictModel

GRAPH_FILE_NAME = "graph_chunk_entity_relation.graphml"
ONE_WAY_PHRASE = "một chiều"


class AlignmentReport(StrictModel):
    """How faithful the LightRAG graph is to the OSM street graph."""

    entity_count: int
    relation_count: int
    graph_is_directed: bool
    entities_matched_to_a_street: int
    entities_matched_to_several_streets: int
    regular_street_count: int
    regular_streets_with_an_entity: int
    relations_between_streets: int
    relations_between_streets_correct: int
    true_regular_street_pairs: int
    true_regular_street_pairs_found: int
    one_way_regular_streets: int
    one_way_regular_streets_described_as_one_way: int
    two_way_regular_streets_described_as_one_way: int
    two_way_regular_street_count: int


def align_lightrag_graph(index_directory: Path, view: GraphView) -> AlignmentReport:
    """Return the alignment of a LightRAG index with the street graph."""
    graph = nx.read_graphml(index_directory / GRAPH_FILE_NAME)
    linker = StreetLinker.from_view(view)
    streets_of_entity = {str(node): linker.resolve(str(node)) for node in graph.nodes}
    matched = {node: ids for node, ids in streets_of_entity.items() if ids}

    regular_ids = {street_id for street_id in view.streets if view.is_regular_street(street_id)}
    true_pairs = {
        tuple(sorted((street_id, other_id)))
        for street_id in regular_ids
        for other_id in view.regular_neighbours(street_id)
    }

    relations_between_streets = 0
    correct_relations = 0
    found_pairs: set[tuple[str, ...]] = set()
    for first, second in graph.edges():
        first_ids = matched.get(str(first), set())
        second_ids = matched.get(str(second), set())
        if not first_ids or not second_ids:
            continue
        relations_between_streets += 1
        candidate_pairs = {tuple(sorted((a, b))) for a in first_ids for b in second_ids if a != b}
        hits = candidate_pairs & true_pairs
        if hits:
            correct_relations += 1
            found_pairs |= hits

    descriptions: dict[str, str] = {}
    for node, ids in matched.items():
        text = str(graph.nodes[node].get("description", "")).lower()
        for street_id in ids:
            descriptions[street_id] = descriptions.get(street_id, "") + " " + text
    one_way_ids = {s for s in regular_ids if view.direction(s) is StreetDirection.ONE_WAY}
    two_way_ids = {
        s
        for s in regular_ids
        if view.direction(s) in (StreetDirection.TWO_WAY, StreetDirection.TWO_WAY_DIVIDED)
    }
    return AlignmentReport(
        entity_count=graph.number_of_nodes(),
        relation_count=graph.number_of_edges(),
        graph_is_directed=graph.is_directed(),
        entities_matched_to_a_street=len(matched),
        entities_matched_to_several_streets=sum(1 for ids in matched.values() if len(ids) > 1),
        regular_street_count=len(regular_ids),
        regular_streets_with_an_entity=len(
            regular_ids & set().union(*matched.values()) if matched else set()
        ),
        relations_between_streets=relations_between_streets,
        relations_between_streets_correct=correct_relations,
        true_regular_street_pairs=len(true_pairs),
        true_regular_street_pairs_found=len(found_pairs),
        one_way_regular_streets=len(one_way_ids),
        one_way_regular_streets_described_as_one_way=sum(
            1 for s in one_way_ids if ONE_WAY_PHRASE in descriptions.get(s, "")
        ),
        two_way_regular_street_count=len(two_way_ids),
        two_way_regular_streets_described_as_one_way=sum(
            1 for s in two_way_ids if ONE_WAY_PHRASE in descriptions.get(s, "")
        ),
    )
