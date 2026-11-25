"""M3: serialize the k-hop neighbourhood of the streets named in a question.

Each line is one street segment between two nodes: "A -> B" is one-way from A to
B, "A <-> B" is two-way. Lines are added in breadth-first order from the named
streets until the character budget is used.

Nodes are named in one of two ways (``NodeNaming``):

- ``LOCAL_IDS`` (M3, EXP-001): short local ids (n1, n2, ...) in order of appearance.
- ``INTERSECTION_NAMES`` (M3N, EXP-002): the names of the streets that meet at the
  node, in square brackets, for example "[Lê Lợi / Pasteur]". Nodes where no named
  street meets get a local id; nodes with the same set of street names are told
  apart by a suffix ("#2").
"""

from __future__ import annotations

from collections import deque
from enum import StrEnum

from graph_llm.questions.graph_view import GraphView
from graph_llm.schemas.question import EdgeRef


class NodeNaming(StrEnum):
    """How nodes are named in the serialized subgraph."""

    LOCAL_IDS = "local_ids"
    INTERSECTION_NAMES = "intersection_names"


HEADER = (
    "Đồ thị con của mạng lưới đường. Mỗi dòng là một đoạn đường nối hai nút giao. "
    "'nA -> nB' là đoạn một chiều, chỉ đi được từ nA đến nB; 'nA <-> nB' là đoạn hai chiều. "
    "Sau đó là tên đường (hoặc '(không tên)'), chiều dài và số làn nếu có."
)


HEADER_INTERSECTION_NAMES = (
    "Đồ thị con của mạng lưới đường. Mỗi dòng là một đoạn đường nối hai nút giao. "
    "Mỗi nút giao được gọi bằng tên các đường gặp nhau tại đó, trong ngoặc vuông, "
    "ví dụ [A / B]; điểm không có đường có tên được gọi là [điểm k]. "
    "'X -> Y' là đoạn một chiều, chỉ đi được từ X đến Y; 'X <-> Y' là đoạn hai chiều. "
    "Sau đó là tên đường của đoạn (hoặc '(không tên)'), chiều dài và số làn nếu có."
)


class NodeNamer:
    """Give every node a name, the first time it is seen."""

    def __init__(self, view: GraphView, naming: NodeNaming) -> None:
        """Create a namer for one context."""
        self.view = view
        self.naming = naming
        self.names: dict[int, str] = {}
        self.uses_of_base_name: dict[str, int] = {}
        self.unnamed_count = 0

    def name(self, node: int) -> str:
        """Return the name of a node."""
        if node not in self.names:
            self.names[node] = self._new_name(node)
        return self.names[node]

    def _new_name(self, node: int) -> str:
        if self.naming is NodeNaming.LOCAL_IDS:
            return f"n{len(self.names) + 1}"
        street_names = sorted(
            {
                self.view.streets[street_id].display_name
                for street_id in self.view.street_ids_at_node.get(node, set())
            }
        )
        if not street_names:
            self.unnamed_count += 1
            return f"[điểm {self.unnamed_count}]"
        base = " / ".join(street_names)
        uses = self.uses_of_base_name.get(base, 0) + 1
        self.uses_of_base_name[base] = uses
        return f"[{base}]" if uses == 1 else f"[{base} #{uses}]"


def hop_distances(view: GraphView, seeds: set[int], hops: int) -> dict[int, int]:
    """Return the hop distance of every node within ``hops`` of the seeds, ignoring direction."""
    distances = dict.fromkeys(seeds, 0)
    queue = deque(sorted(seeds))
    while queue:
        node = queue.popleft()
        if distances[node] == hops:
            continue
        neighbours = set(view.graph.successors(node)) | set(view.graph.predecessors(node))
        for neighbour in sorted(neighbours):
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return distances


def edge_label(view: GraphView, edge: EdgeRef) -> str:
    """Return the street name, length and lane count of an edge."""
    owners = view.street_ids_of_edge.get(edge, [])
    name = " / ".join(sorted(view.streets[o].display_name for o in owners)) or "(không tên)"
    label = f"{name} | {round(view.edge_length_m(edge))} m"
    lanes = view.edge_attributes(edge).get("lanes")
    if lanes is not None and not isinstance(lanes, list):
        label += f" | {lanes} làn"
    return label


def build_subgraph_context(
    view: GraphView,
    street_ids: set[str],
    hops: int,
    budget_characters: int,
    naming: NodeNaming = NodeNaming.LOCAL_IDS,
) -> tuple[str, set[EdgeRef]]:
    """Return the serialized neighbourhood of the given streets and the edges it contains."""
    seeds = set().union(*(view.nodes_of_street[street_id] for street_id in street_ids))
    if not seeds:
        return "", set()
    distances = hop_distances(view, seeds, hops)
    pairs: dict[tuple[int, int], list[EdgeRef]] = {}
    for source, target, key in view.graph.edges(keys=True):
        if source in distances and target in distances:
            pair = (min(source, target), max(source, target))
            pairs.setdefault(pair, []).append(
                EdgeRef(source_node_id=source, target_node_id=target, key=key)
            )
    ordered = sorted(
        pairs.items(),
        key=lambda item: (max(distances[item[0][0]], distances[item[0][1]]), item[0]),
    )
    namer = NodeNamer(view, naming)
    local = namer.name
    header = HEADER if naming is NodeNaming.LOCAL_IDS else HEADER_INTERSECTION_NAMES
    lines = [header]
    used = len(header)
    covered: set[EdgeRef] = set()
    for (first, second), edges in ordered:
        forward = [edge for edge in edges if edge.source_node_id == first]
        backward = [edge for edge in edges if edge.source_node_id == second]
        new_lines: list[str] = []
        if forward and backward and edge_label(view, forward[0]) == edge_label(view, backward[0]):
            new_lines.append(f"{local(first)} <-> {local(second)} | {edge_label(view, forward[0])}")
        else:
            for edge in forward + backward:
                arrow = f"{local(edge.source_node_id)} -> {local(edge.target_node_id)}"
                new_lines.append(f"{arrow} | {edge_label(view, edge)}")
        added = sum(len(line) + 1 for line in new_lines)
        if used + added > budget_characters:
            break
        lines += new_lines
        used += added
        covered.update(edges)
    return "\n".join(lines), covered
