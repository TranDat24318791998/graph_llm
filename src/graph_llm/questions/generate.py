"""Generate benchmark questions from the processed graph.

Each template has a candidate function that lists every valid question for it,
with the answer and the required evidence computed from the graph. Sampling then
draws the configured number of candidates with a fixed seed.

Definitions shared by all templates:

- A question names only askable streets: regular streets whose name is unambiguous.
- "Streets that meet street S" are the other regular streets that share a node with
  S. Alleys, internal roads and unnamed edges are never counted.
- An intersection of two streets is the set of nodes they share, used only if
  those nodes lie within 80 m of each other (one place, possibly several nodes
  for divided roads).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from graph_llm.graph.carriageways import StreetDirection
from graph_llm.questions.graph_view import GraphView
from graph_llm.questions.routes import (
    route_street_sequence,
    shortest_length_ignoring_direction,
    shortest_route,
)
from graph_llm.schemas.benchmark import BenchmarkConfig, QuestionTemplate
from graph_llm.schemas.question import (
    Answer,
    BooleanAnswer,
    EdgeRef,
    Evidence,
    NumberAnswer,
    Question,
    QuestionFeatures,
    QuestionType,
    RouteAnswer,
    StreetNameAnswer,
    StreetSetAnswer,
)

L2_MIN_ANSWER_SIZE = 2
L2_MAX_ANSWER_SIZE = 8
L4_MIN_LENGTH_M = 400.0
L4_MAX_LENGTH_M = 2000.0
L4_MIN_STREETS = 2
L4_MAX_STREETS = 5
L4_CANDIDATE_PAIRS = 4000
DIRECTION_DEPENDENCE_RATIO = 1.01
LENGTH_QUESTION_MIN_M = 150.0
LENGTH_THRESHOLD_STEP_M = 50
COUNT_STRATUM_TOP = 4
LENGTH_MARGIN_SHARE = 0.25


@dataclass(frozen=True)
class Candidate:
    """A valid question before sampling: slots, answer, evidence and features."""

    slots: dict[str, str]
    answer: Answer
    evidence: Evidence
    features: QuestionFeatures
    stratum: str = ""


def make_evidence(view: GraphView, edges: list[EdgeRef]) -> Evidence:
    """Return evidence made of unique edges and their end nodes, in a stable order."""
    unique_edges = sorted(
        set(edges), key=lambda edge: (edge.source_node_id, edge.target_node_id, edge.key)
    )
    nodes = sorted(
        {node for edge in unique_edges for node in (edge.source_node_id, edge.target_node_id)}
    )
    return Evidence(node_ids=nodes, edges=unique_edges)


def meeting_edges(view: GraphView, street_id: str, other_ids: dict[str, set[int]]) -> list[EdgeRef]:
    """Return the edges of a street and of its neighbours at the nodes where they meet."""
    edges: list[EdgeRef] = []
    for other_id, nodes in other_ids.items():
        for node in nodes:
            edges += view.incident_street_edges(node, street_id)
            edges += view.incident_street_edges(node, other_id)
    return edges


def askable_street_ids(view: GraphView) -> list[str]:
    """Return the askable streets in a stable order."""
    return sorted(street_id for street_id in view.streets if view.is_askable(street_id))


def name_of(view: GraphView, street_id: str) -> str:
    """Return the display name of a street, as used in answers."""
    return view.streets[street_id].display_name


def slot_name(view: GraphView, street_id: str) -> str:
    """Return a street name for a template slot that is already preceded by "đường".

    A display name that starts with "Đường" (for example "Đường 10A") loses that
    word, so the question does not read "đường Đường 10A".
    """
    display_name = name_of(view, street_id)
    prefix = "Đường "
    return display_name[len(prefix) :] if display_name.startswith(prefix) else display_name


def l1_oneway_candidates(view: GraphView) -> list[Candidate]:
    """Is street S one-way? Only streets that are clearly one-way or clearly two-way.

    A divided road (two carriageways with a median) is two-way. Streets with mixed
    one-way and two-way parts are not asked about.
    """
    candidates = []
    for street_id in askable_street_ids(view):
        status = view.direction(street_id)
        if status is StreetDirection.MIXED:
            continue
        candidates.append(
            Candidate(
                slots={"street": slot_name(view, street_id)},
                answer=BooleanAnswer(value=status is StreetDirection.ONE_WAY),
                evidence=make_evidence(view, view.streets[street_id].edges),
                features=QuestionFeatures(
                    hop_count=0, requires_computation=False, depends_on_edge_direction=True
                ),
                # Balance yes and no answers; divided roads fall among the "no" answers.
                stratum="yes" if status is StreetDirection.ONE_WAY else "no",
            )
        )
    return candidates


def l1_length_candidates(view: GraphView, rng: random.Random) -> list[Candidate]:
    """Is street S longer than a threshold? Thresholds are at least 25% away from the length.

    Half of the streets get a threshold below their length (answer yes), half
    above (answer no), so a constant guess scores 50%.
    """
    candidates = []
    for index, street_id in enumerate(askable_street_ids(view)):
        length_m = view.street_length_m(street_id)
        if length_m < LENGTH_QUESTION_MIN_M:
            continue
        is_longer = index % 2 == 0
        factor = rng.uniform(0.5, 0.75) if is_longer else rng.uniform(1.33, 2.0)
        threshold_m = max(LENGTH_THRESHOLD_STEP_M, round(length_m * factor / 50) * 50)
        if abs(length_m - threshold_m) < LENGTH_MARGIN_SHARE * length_m:
            continue
        candidates.append(
            Candidate(
                slots={"street": slot_name(view, street_id), "threshold_m": str(threshold_m)},
                answer=BooleanAnswer(value=length_m > threshold_m),
                evidence=make_evidence(view, view.streets[street_id].edges),
                features=QuestionFeatures(
                    hop_count=0, requires_computation=True, depends_on_edge_direction=False
                ),
                stratum=str(length_m > threshold_m),
            )
        )
    return candidates


def value_stratum(value: float, top: int) -> str:
    """Return the stratum of a count answer: its value, with all values from ``top`` pooled."""
    return f"{top}+" if value >= top else str(int(value))


def l1_lanes_candidates(view: GraphView) -> list[Candidate]:
    """How many lanes does street S have? Only streets with one stated value over 90% of length."""
    candidates = []
    for street_id in askable_street_ids(view):
        lanes = view.lane_count(street_id)
        if lanes is None or view.direction(street_id) is StreetDirection.MIXED:
            continue
        candidates.append(
            Candidate(
                slots={"street": slot_name(view, street_id)},
                answer=NumberAnswer(value=float(lanes)),
                evidence=make_evidence(view, view.streets[street_id].edges),
                features=QuestionFeatures(
                    hop_count=0, requires_computation=False, depends_on_edge_direction=False
                ),
            )
        )
    return candidates


def l2_neighbours_candidates(view: GraphView) -> list[Candidate]:
    """Which streets meet street S?"""
    candidates = []
    for street_id in askable_street_ids(view):
        neighbours = view.regular_neighbours(street_id)
        names = sorted({name_of(view, other_id) for other_id in neighbours})
        if not L2_MIN_ANSWER_SIZE <= len(names) <= L2_MAX_ANSWER_SIZE:
            continue
        candidates.append(
            Candidate(
                slots={"street": slot_name(view, street_id)},
                answer=StreetSetAnswer(street_names=names),
                evidence=make_evidence(view, meeting_edges(view, street_id, neighbours)),
                features=QuestionFeatures(
                    hop_count=1, requires_computation=False, depends_on_edge_direction=False
                ),
            )
        )
    return candidates


def walk_to_next_street(
    view: GraphView, oneway_id: str, start_nodes: set[int], excluded: set[str], forward: bool
) -> tuple[str, list[EdgeRef], set[int]] | None:
    """Walk along a one-way street out of a junction until the next junction with another street.

    Args:
        view: The graph.
        oneway_id: The street walked along.
        start_nodes: The nodes of the junction the walk starts from.
        excluded: Streets that do not count as met (the crossing street).
        forward: Walk in the street's direction of travel (True) or against it.

    Returns:
        The street met, the edges walked, and the nodes of the junction where it is met;
        None if the street leaves the start junction in several ways, ends first, or meets
        two streets at the same junction.
    """
    next_edge: dict[int, EdgeRef] = {}
    exits = []
    for edge in view.streets[oneway_id].edges:
        tail, head = (
            (edge.source_node_id, edge.target_node_id)
            if forward
            else (edge.target_node_id, edge.source_node_id)
        )
        next_edge[tail] = edge
        if tail in start_nodes and head not in start_nodes:
            exits.append(edge)
    if len(exits) != 1:
        return None
    walked: list[EdgeRef] = []
    edge = exits[0]
    while True:
        walked.append(edge)
        node = edge.target_node_id if forward else edge.source_node_id
        met = view.regular_street_ids_at(node) - excluded - {oneway_id}
        junction_id = view.junction_id_of_node.get(node)
        nodes = set(view.junctions[junction_id].node_ids) if junction_id else {node}
        if len(met) == 1:
            return met.pop(), walked, nodes
        if len(met) > 1 or node not in next_edge or len(walked) > len(next_edge):
            return None
        edge = next_edge[node]


def junction_street_edges(view: GraphView, nodes: set[int], street_id: str) -> list[EdgeRef]:
    """Return the edges of a street that touch any node of a junction."""
    return [
        edge
        for edge in view.streets[street_id].edges
        if edge.source_node_id in nodes or edge.target_node_id in nodes
    ]


def l3_next_on_oneway_candidates(view: GraphView) -> list[Candidate]:
    """At the intersection of X and one-way street Y, which street does Y meet next?"""
    candidates = []
    for oneway_id in askable_street_ids(view):
        if view.direction(oneway_id) is not StreetDirection.ONE_WAY:
            continue
        for cross_id in sorted(view.regular_neighbours(oneway_id)):
            if not view.is_askable(cross_id):
                continue
            start_nodes = view.clean_intersection(oneway_id, cross_id)
            if start_nodes is None:
                continue
            forward = walk_to_next_street(view, oneway_id, start_nodes, {cross_id}, forward=True)
            if forward is None:
                continue
            answer_id, walked, end_nodes = forward
            backward = walk_to_next_street(view, oneway_id, start_nodes, {cross_id}, forward=False)
            depends_on_direction = backward is None or name_of(view, backward[0]) != name_of(
                view, answer_id
            )
            edges = walked + junction_street_edges(view, start_nodes, cross_id)
            edges += junction_street_edges(view, end_nodes, answer_id)
            candidates.append(
                Candidate(
                    slots={
                        "cross_street": slot_name(view, cross_id),
                        "oneway_street": slot_name(view, oneway_id),
                    },
                    answer=StreetNameAnswer(street_name=name_of(view, answer_id)),
                    evidence=make_evidence(view, edges),
                    features=QuestionFeatures(
                        hop_count=len(walked),
                        requires_computation=False,
                        depends_on_edge_direction=depends_on_direction,
                    ),
                )
            )
    return candidates


def clean_intersections(view: GraphView) -> list[tuple[str, str, set[int]]]:
    """Return every pair of askable streets that meet in one place, with its nodes."""
    intersections = []
    for street_id in askable_street_ids(view):
        for other_id in sorted(view.regular_neighbours(street_id)):
            if other_id <= street_id or not view.is_askable(other_id):
                continue
            nodes = view.clean_intersection(street_id, other_id)
            if nodes is not None:
                intersections.append((street_id, other_id, nodes))
    return intersections


def l4_shortest_route_candidates(view: GraphView, rng: random.Random) -> list[Candidate]:
    """Shortest legal route between two intersections, as street sequence and length."""
    intersections = clean_intersections(view)
    candidates = []
    seen: set[tuple[int, int]] = set()
    for _ in range(L4_CANDIDATE_PAIRS):
        start_index, end_index = rng.sample(range(len(intersections)), 2)
        if (start_index, end_index) in seen:
            continue
        seen.add((start_index, end_index))
        start_a, start_b, start_nodes = intersections[start_index]
        end_a, end_b, end_nodes = intersections[end_index]
        if start_nodes & end_nodes:
            continue
        route = shortest_route(view, start_nodes, end_nodes)
        if route is None:
            continue
        length_m = route.length_m
        if not L4_MIN_LENGTH_M <= length_m <= L4_MAX_LENGTH_M:
            continue
        if view.timed_rules.route_violates(route.edges):
            # Decision Q-B (ii): no route question whose answer depends on the time of day.
            continue
        sequence = route_street_sequence(view, route.edges)
        if sequence is None or not L4_MIN_STREETS <= len(sequence) <= L4_MAX_STREETS:
            continue
        undirected_m = shortest_length_ignoring_direction(view, start_nodes, end_nodes)
        depends_on_direction = (
            undirected_m is not None and length_m > undirected_m * DIRECTION_DEPENDENCE_RATIO
        )
        path_edges = route.edges
        node_path = route.nodes
        candidates.append(
            Candidate(
                slots={
                    "start_a": slot_name(view, start_a),
                    "start_b": slot_name(view, start_b),
                    "end_a": slot_name(view, end_a),
                    "end_b": slot_name(view, end_b),
                    "start_node_ids": ",".join(str(node) for node in sorted(start_nodes)),
                    "end_node_ids": ",".join(str(node) for node in sorted(end_nodes)),
                },
                answer=RouteAnswer(street_names=sequence, length_m=length_m, node_path=node_path),
                evidence=make_evidence(view, path_edges),
                features=QuestionFeatures(
                    hop_count=len(path_edges),
                    requires_computation=True,
                    depends_on_edge_direction=depends_on_direction,
                ),
            )
        )
    return candidates


def l5_two_way_neighbours_candidates(view: GraphView) -> list[Candidate]:
    """Among the streets that meet S, how many are two-way along their whole length?

    Divided roads count as two-way.
    """
    candidates = []
    for street_id in askable_street_ids(view):
        neighbours = view.regular_neighbours(street_id)
        names = {name_of(view, other_id) for other_id in neighbours}
        if not L2_MIN_ANSWER_SIZE <= len(names) <= L2_MAX_ANSWER_SIZE:
            continue
        if len(names) != len(neighbours):
            continue
        two_way = [
            other_id
            for other_id in neighbours
            if view.direction(other_id)
            in (StreetDirection.TWO_WAY, StreetDirection.TWO_WAY_DIVIDED)
        ]
        edges = meeting_edges(view, street_id, neighbours)
        for other_id in neighbours:
            edges += view.streets[other_id].edges
        candidates.append(
            Candidate(
                slots={"street": slot_name(view, street_id)},
                answer=NumberAnswer(value=float(len(two_way))),
                evidence=make_evidence(view, edges),
                stratum=value_stratum(len(two_way), COUNT_STRATUM_TOP),
                features=QuestionFeatures(
                    hop_count=1, requires_computation=True, depends_on_edge_direction=True
                ),
            )
        )
    return candidates


def l5_common_neighbours_candidates(view: GraphView) -> list[Candidate]:
    """How many streets meet both S and T? Only pairs that do not meet each other."""
    neighbour_map = {
        street_id: view.regular_neighbours(street_id) for street_id in askable_street_ids(view)
    }
    candidates = []
    for street_id, neighbours in neighbour_map.items():
        second_hop = {
            far_id
            for other_id in neighbours
            for far_id in view.regular_neighbours(other_id)
            if far_id in neighbour_map and far_id > street_id and far_id not in neighbours
        }
        for far_id in sorted(second_hop):
            common = set(neighbours) & set(neighbour_map[far_id])
            if not common:
                continue
            edges = meeting_edges(view, street_id, {other: neighbours[other] for other in common})
            edges += meeting_edges(
                view, far_id, {other: neighbour_map[far_id][other] for other in common}
            )
            common_count = len({name_of(view, other) for other in common})
            candidates.append(
                Candidate(
                    slots={
                        "street_a": slot_name(view, street_id),
                        "street_b": slot_name(view, far_id),
                    },
                    answer=NumberAnswer(value=float(common_count)),
                    stratum=value_stratum(common_count, COUNT_STRATUM_TOP),
                    evidence=make_evidence(view, edges),
                    features=QuestionFeatures(
                        hop_count=2, requires_computation=True, depends_on_edge_direction=False
                    ),
                )
            )
    return candidates


QUESTION_TYPE_OF_TEMPLATE: dict[str, QuestionType] = {
    "L1_oneway": QuestionType.L1_ATTRIBUTE,
    "L1_lanes": QuestionType.L1_ATTRIBUTE,
    "L1_length": QuestionType.L1_ATTRIBUTE,
    "L2_neighbours": QuestionType.L2_ADJACENCY,
    "L3_next_on_oneway": QuestionType.L3_MULTI_STEP,
    "L4_shortest_route": QuestionType.L4_ROUTE,
    "L5_two_way_neighbours": QuestionType.L5_AGGREGATION,
    "L5_common_neighbours": QuestionType.L5_AGGREGATION,
}


def candidates_for(template_id: str, view: GraphView, rng: random.Random) -> list[Candidate]:
    """Return all candidates of one template."""
    simple: dict[str, Callable[[GraphView], list[Candidate]]] = {
        "L1_oneway": l1_oneway_candidates,
        "L1_lanes": l1_lanes_candidates,
        "L2_neighbours": l2_neighbours_candidates,
        "L3_next_on_oneway": l3_next_on_oneway_candidates,
        "L5_two_way_neighbours": l5_two_way_neighbours_candidates,
        "L5_common_neighbours": l5_common_neighbours_candidates,
    }
    if template_id == "L4_shortest_route":
        return l4_shortest_route_candidates(view, rng)
    if template_id == "L1_length":
        return l1_length_candidates(view, rng)
    return simple[template_id](view)


def sample_candidates(
    candidates: list[Candidate], count: int, rng: random.Random
) -> list[Candidate]:
    """Draw candidates, balanced across strata when the template defines strata.

    Raises:
        ValueError: If there are fewer candidates than requested.
    """
    strata = sorted({candidate.stratum for candidate in candidates})
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} candidates for {count} questions.")
    if len(strata) == 1:
        return rng.sample(candidates, count)
    chosen: list[Candidate] = []
    per_stratum = count // len(strata)
    for stratum in strata:
        members = [candidate for candidate in candidates if candidate.stratum == stratum]
        chosen += rng.sample(members, min(per_stratum, len(members)))
    remaining = [candidate for candidate in candidates if candidate not in chosen]
    chosen += rng.sample(remaining, count - len(chosen))
    return chosen


def generate_questions(view: GraphView, config: BenchmarkConfig) -> list[Question]:
    """Generate all template questions of a benchmark version, before paraphrasing."""
    rng = random.Random(config.seed)
    questions: list[Question] = []
    for template in config.templates:
        candidates = candidates_for(template.template_id, view, rng)
        for number, candidate in enumerate(
            sample_candidates(candidates, template.count, rng), start=1
        ):
            questions.append(build_question(template, candidate, number, config))
    return questions


def build_question(
    template: QuestionTemplate, candidate: Candidate, number: int, config: BenchmarkConfig
) -> Question:
    """Fill a template with a candidate."""
    template_text = template.text.format(**candidate.slots)
    return Question(
        question_id=f"{template.template_id}_{number:03d}",
        benchmark_version=config.benchmark_version,
        question_type=QUESTION_TYPE_OF_TEMPLATE[template.template_id],
        template_id=template.template_id,
        template_text=template_text,
        text=template_text,
        slots=candidate.slots,
        answer=candidate.answer,
        evidence=candidate.evidence,
        features=candidate.features,
    )
