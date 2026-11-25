"""M3T: corpus text selected by the street graph instead of by embedding similarity.

The streets named in the question are found with ``StreetLinker`` (as in M3).
Their street documents are put into the context first, then the documents of the
streets that meet them, then of the streets that meet those, and so on up to
``hops`` street-level hops. Within one hop, regular streets come before alleys
and internal roads, and streets are ordered by name. Whole chunks are added in
that order while they fit the character budget.

The text is the same template-generated corpus that M1 retrieves from, so M1 and
M3T differ only in how the text is selected.
"""

from __future__ import annotations

from collections import defaultdict

from graph_llm.questions.graph_view import GraphView
from graph_llm.representations.corpus_store import Corpus
from graph_llm.schemas.corpus import TextChunk
from graph_llm.schemas.question import EdgeRef
from graph_llm.schemas.street import StreetCategory

CHUNK_SEPARATOR = "\n\n"


def meeting_street_ids(view: GraphView, street_id: str) -> set[str]:
    """Return every street, of any category, that shares a node with a street."""
    return {
        other_id
        for node in view.nodes_of_street[street_id]
        for other_id in view.street_ids_at_node.get(node, set())
        if other_id != street_id
    }


def streets_in_hop_order(view: GraphView, seed_ids: set[str], hops: int) -> list[str]:
    """Return the seed streets and their street-level neighbourhood in reading order."""
    ordered: list[str] = []
    seen: set[str] = set()
    frontier = set(seed_ids)
    for _ in range(hops + 1):
        layer = sorted(
            frontier - seen,
            key=lambda s: (
                view.streets[s].category is not StreetCategory.REGULAR,
                view.streets[s].display_name,
                s,
            ),
        )
        ordered += layer
        seen |= set(layer)
        frontier = set().union(*(meeting_street_ids(view, s) for s in layer)) if layer else set()
    return ordered


def graph_linked_chunks(
    view: GraphView,
    corpus: Corpus,
    street_ids: set[str],
    hops: int,
    budget_characters: int,
) -> tuple[str, set[EdgeRef]]:
    """Return the chunks of the linked streets and their neighbours, joined, and their edges."""
    chunks_of_street: dict[str, list[TextChunk]] = defaultdict(list)
    street_of_document = {document.document_id: document.street_id for document in corpus.documents}
    for chunk in corpus.chunks:
        chunks_of_street[street_of_document[chunk.document_id]].append(chunk)
    texts: list[str] = []
    used = 0
    covered: set[EdgeRef] = set()
    for street_id in streets_in_hop_order(view, street_ids, hops):
        for chunk in chunks_of_street.get(street_id, []):
            cost = len(chunk.text) + len(CHUNK_SEPARATOR)
            if used + cost > budget_characters:
                continue
            texts.append(chunk.text)
            used += cost
            covered.update(chunk.edges)
    return CHUNK_SEPARATOR.join(texts), covered
