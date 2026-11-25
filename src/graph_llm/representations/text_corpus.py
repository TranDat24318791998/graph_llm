"""Generate the text corpus that M1 and M2 read: one description per named street.

The text is produced from the processed graph by fixed sentence templates, so it
states only facts of the graph. Each street document has a header (category,
direction of travel, lane count when known, the streets it meets) and one
sentence per segment between two consecutive points where another named street
meets it. Unnamed edges have no text.

Every chunk records which graph edges it describes. This is what evidence recall
is measured against for the text representations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from graph_llm.graph.carriageways import EdgeDirection, StreetDirection
from graph_llm.questions.graph_view import GraphView
from graph_llm.schemas.corpus import TextChunk, TextDocument
from graph_llm.schemas.question import EdgeRef
from graph_llm.schemas.street import StreetCategory

CHUNK_CHARACTER_LIMIT = 1500

CATEGORY_PHRASES: dict[StreetCategory, str] = {
    StreetCategory.REGULAR: "là một tuyến đường",
    StreetCategory.ALLEY: "là một con hẻm",
    StreetCategory.INTERNAL: "là một đường nội bộ trong khu dân cư",
}

PLACE_PHRASE = "là một quảng trường hoặc vòng xoay, nơi các tuyến đường gặp nhau"

DIRECTION_PHRASES: dict[StreetDirection, str] = {
    StreetDirection.ONE_WAY: "Toàn tuyến là đường một chiều.",
    StreetDirection.TWO_WAY: "Toàn tuyến cho phép xe đi hai chiều.",
    StreetDirection.TWO_WAY_DIVIDED: (
        "Toàn tuyến cho phép xe đi hai chiều; có đoạn là đường đôi có dải phân cách, "
        "mỗi chiều đi trên một bên đường riêng."
    ),
    StreetDirection.MIXED: (
        "Tuyến đường có đoạn một chiều và đoạn hai chiều; chi tiết ở từng đoạn dưới đây."
    ),
}


@dataclass(frozen=True)
class Segment:
    """A directed walk along one street between two key nodes."""

    start: int
    end: int
    edges: list[EdgeRef]


def streets_meeting_at(view: GraphView, node: int, street_id: str) -> set[str]:
    """Return the other named streets and places that reach the junction of a node."""
    junction_id = view.junction_id_of_node.get(node)
    if junction_id is None:
        others = set(view.street_ids_at_node.get(node, set()))
    else:
        junction = view.junctions[junction_id]
        others = set(junction.street_ids) | {
            other for other in view.street_ids_at_node.get(node, set()) if view.is_place(other)
        }
    return others - {street_id}


def key_nodes(view: GraphView, street_id: str) -> set[int]:
    """Return the nodes where another named street meets a street, or where it ends or forks."""
    neighbours: dict[int, set[int]] = defaultdict(set)
    for edge in view.streets[street_id].edges:
        neighbours[edge.source_node_id].add(edge.target_node_id)
        neighbours[edge.target_node_id].add(edge.source_node_id)
    return {
        node
        for node, adjacent in neighbours.items()
        if len(adjacent) != 2 or streets_meeting_at(view, node, street_id)
    }


def directed_segments(view: GraphView, street_id: str) -> list[Segment]:
    """Split the edges of a street into directed walks between key nodes."""
    keys = key_nodes(view, street_id)
    outgoing: dict[int, list[EdgeRef]] = defaultdict(list)
    for edge in view.streets[street_id].edges:
        outgoing[edge.source_node_id].append(edge)
    segments = []
    used: set[EdgeRef] = set()
    for start in sorted(keys):
        for first_edge in outgoing[start]:
            if first_edge in used:
                continue
            edges = [first_edge]
            used.add(first_edge)
            previous, node = start, first_edge.target_node_id
            while node not in keys:
                onward = [
                    edge
                    for edge in outgoing[node]
                    if edge.target_node_id != previous and edge not in used
                ]
                if len(onward) != 1:
                    break
                edges.append(onward[0])
                used.add(onward[0])
                previous, node = node, onward[0].target_node_id
            segments.append(Segment(start=start, end=node, edges=edges))
    return segments


def node_label(view: GraphView, node: int, street_id: str) -> str:
    """Describe a point of a street by the other named streets that meet it at its junction."""
    others = sorted(
        {
            view.streets[other_id].display_name
            for other_id in streets_meeting_at(view, node, street_id)
        }
    )
    if not others:
        return "điểm không có đường có tên giao cắt"
    return "giao lộ " + " / ".join(others)


def within_one_junction(view: GraphView, segment: Segment) -> bool:
    """Return True if a segment starts and ends in the same junction."""
    start = view.junction_id_of_node.get(segment.start)
    return start is not None and start == view.junction_id_of_node.get(segment.end)


def segment_sentences(view: GraphView, street_id: str) -> list[tuple[str, list[EdgeRef]]]:
    """Return one sentence per segment, merging the two directions of a two-way segment."""
    name = view.streets[street_id].display_name
    segments = directed_segments(view, street_id)
    by_ends = {(segment.start, segment.end): segment for segment in segments}
    sentences = []
    described: set[tuple[int, int]] = set()
    for segment in segments:
        if (segment.start, segment.end) in described:
            continue
        if within_one_junction(view, segment):
            # A piece inside a junction (median crossing, roundabout ring) is part of the
            # intersection, not a stretch of street between two intersections.
            continue
        described.add((segment.start, segment.end))
        length_m = round(sum(view.edge_length_m(edge) for edge in segment.edges))
        start_label = node_label(view, segment.start, street_id)
        end_label = node_label(view, segment.end, street_id)
        reverse = by_ends.get((segment.end, segment.start))
        if reverse is not None:
            described.add((segment.end, segment.start))
            sentences.append(
                (
                    f"Đoạn {name} giữa {start_label} và {end_label}: {length_m} m, hai chiều.",
                    segment.edges + reverse.edges,
                )
            )
        else:
            labels = view.edge_labels[street_id]
            divided = all(labels[edge] is EdgeDirection.DIVIDED for edge in segment.edges)
            kind = "một bên của đường đôi, một chiều" if divided else "một chiều"
            sentences.append(
                (
                    f"Đoạn {name} từ {start_label} đến {end_label}: {length_m} m, "
                    f"{kind} (xe chỉ được đi từ {start_label} về {end_label}).",
                    segment.edges,
                )
            )
    return sentences


def street_header(view: GraphView, street_id: str) -> str:
    """Return the summary sentences of a street."""
    street = view.streets[street_id]
    if view.is_place(street_id):
        parts = [f"{street.display_name} {PLACE_PHRASE}."]
    else:
        parts = [f"Đường {street.display_name} {CATEGORY_PHRASES[street.category]}."]
    if street.aliases:
        parts.append(f"Tên gọi khác: {', '.join(street.aliases)}.")
    if not view.is_place(street_id):
        parts.append(DIRECTION_PHRASES[view.direction(street_id)])
    if view.is_place(street_id):
        meeting = sorted(
            {
                view.streets[other_id].display_name
                for node in view.nodes_of_street[street_id]
                for other_id in streets_meeting_at(view, node, street_id)
            }
        )
        if meeting:
            parts.append("Các đường gặp nhau tại đây: " + ", ".join(meeting) + ".")
        return " ".join(parts)
    lanes = view.lane_count(street_id)
    if lanes is not None:
        parts.append(f"Đường có {lanes} làn xe.")
    neighbour_names = sorted(
        {
            view.streets[other_id].display_name
            for node in view.nodes_of_street[street_id]
            for other_id in streets_meeting_at(view, node, street_id)
        }
    )
    if neighbour_names:
        parts.append(
            f"Đường {street.display_name} giao nhau hoặc nối trực tiếp với: "
            + ", ".join(neighbour_names)
            + "."
        )
    return " ".join(parts)


def build_street_document(view: GraphView, street_id: str) -> tuple[TextDocument, list[TextChunk]]:
    """Return the document of one street and its chunks.

    The first chunk holds the header and covers every edge of the street, because
    the header summarizes all of them. Further chunks start with the street name
    so they can be read on their own.
    """
    street = view.streets[street_id]
    header = street_header(view, street_id)
    sentences = segment_sentences(view, street_id)
    chunk_texts: list[str] = [header]
    chunk_edges: list[list[EdgeRef]] = [list(street.edges)]
    for sentence, edges in sentences:
        if len(chunk_texts[-1]) + len(sentence) + 1 > CHUNK_CHARACTER_LIMIT:
            chunk_texts.append(f"Đường {street.display_name} (tiếp theo).")
            chunk_edges.append([])
        chunk_texts[-1] += " " + sentence
        chunk_edges[-1] += edges
    document_id = f"street-{street_id}"
    document = TextDocument(
        document_id=document_id, street_id=street_id, text="\n".join(chunk_texts)
    )
    chunks = [
        TextChunk(
            chunk_id=f"{document_id}-{index}",
            document_id=document_id,
            text=text,
            edges=sorted(
                set(edges), key=lambda edge: (edge.source_node_id, edge.target_node_id, edge.key)
            ),
        )
        for index, (text, edges) in enumerate(zip(chunk_texts, chunk_edges, strict=True))
    ]
    return document, chunks


def build_corpus(view: GraphView) -> tuple[list[TextDocument], list[TextChunk]]:
    """Return the documents and chunks of every named street, in a stable order."""
    documents: list[TextDocument] = []
    chunks: list[TextChunk] = []
    for street_id in sorted(view.streets):
        document, street_chunks = build_street_document(view, street_id)
        documents.append(document)
        chunks += street_chunks
    return documents, chunks
