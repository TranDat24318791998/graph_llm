"""Score a parsed answer against the ground truth. No LLM is involved.

- Boolean, number (L1, L5) and single street name (L3): exact match.
- Street set (L2): set F1 over resolved streets; correct only when F1 is 1.
- Route (L4): the predicted street sequence is checked on the graph. It is
  correct when a legal route exists that follows the sequence in order and is at
  most 2% longer than the true shortest route. Reported separately: legality,
  wrong direction (a route exists only against one-way direction), and whether the
  stated length is within 10% of the true length.
"""

from __future__ import annotations

from graph_llm.names.normalize import ascii_key, name_key
from graph_llm.questions.graph_view import GraphView
from graph_llm.questions.routes import shortest_length_following_sequence
from graph_llm.representations.linker import StreetLinker
from graph_llm.schemas.question import (
    Answer,
    BooleanAnswer,
    NumberAnswer,
    Question,
    RouteAnswer,
    StreetNameAnswer,
    StreetSetAnswer,
)

ROUTE_OPTIMALITY_TOLERANCE = 0.02
ROUTE_LENGTH_TOLERANCE = 0.10


def same_street_name(predicted: str, expected: str) -> bool:
    """Return True if two names match by name key, or by ASCII key if diacritics were dropped."""
    return name_key(predicted) == name_key(expected) or ascii_key(predicted) == ascii_key(expected)


def set_f1(predicted: list[str], expected: list[str]) -> float:
    """Return the F1 of two name sets, matching names with ``same_street_name``."""
    by_ascii_key = {ascii_key(name): name_key(name) for name in reversed(predicted)}
    unique_predicted = list(by_ascii_key.values())
    expected_keys = [name_key(name) for name in expected]
    if not unique_predicted and not expected_keys:
        return 1.0
    matched = sum(
        1
        for expected_name in expected_keys
        if any(same_street_name(guess, expected_name) for guess in unique_predicted)
    )
    if matched == 0:
        return 0.0
    precision = matched / len(unique_predicted)
    recall = matched / len(expected_keys)
    return 2 * precision * recall / (precision + recall)


def node_set(slot: str) -> set[int]:
    """Parse a comma-separated node id slot."""
    return {int(node) for node in slot.split(",") if node}


def score_route(
    view: GraphView, linker: StreetLinker, question: Question, predicted: RouteAnswer
) -> tuple[bool, dict[str, float]]:
    """Check a predicted route on the graph. Returns correctness and the sub-scores."""
    expected = question.answer
    assert isinstance(expected, RouteAnswer)
    sources = node_set(question.slots["start_node_ids"])
    targets = node_set(question.slots["end_node_ids"])
    stages = [linker.resolve(name) for name in predicted.street_names]
    known = bool(stages) and all(stages)
    legal_length = (
        shortest_length_following_sequence(view, sources, targets, stages) if known else None
    )
    any_direction_length = (
        shortest_length_following_sequence(view, sources, targets, stages, respect_rules=False)
        if known
        else None
    )
    legal = legal_length is not None
    optimal = legal_length is not None and legal_length <= expected.length_m * (
        1 + ROUTE_OPTIMALITY_TOLERANCE
    )
    sub_scores = {
        "all_streets_known": float(known),
        "legal": float(legal),
        # The streets can be followed in this order, but only against an edge direction
        # or through a forbidden turn (graph v2 routes obey turn restrictions).
        "breaks_direction_or_turn_rule": float(not legal and any_direction_length is not None),
        "optimal": float(optimal),
        "length_within_10_percent": float(
            abs(predicted.length_m - expected.length_m)
            <= ROUTE_LENGTH_TOLERANCE * expected.length_m
        ),
        "sequence_exact": float(
            len(predicted.street_names) == len(expected.street_names)
            and all(
                same_street_name(guess, truth)
                for guess, truth in zip(predicted.street_names, expected.street_names, strict=True)
            )
        ),
    }
    return optimal, sub_scores


def score_answer(
    view: GraphView, linker: StreetLinker, question: Question, predicted: Answer | None
) -> tuple[bool, float, dict[str, float]]:
    """Return (is_correct, score in [0, 1], sub_scores) for one prediction."""
    expected = question.answer
    if predicted is None or predicted.kind != expected.kind:
        return False, 0.0, {}
    if isinstance(expected, BooleanAnswer) and isinstance(predicted, BooleanAnswer):
        correct = predicted.value == expected.value
    elif isinstance(expected, NumberAnswer) and isinstance(predicted, NumberAnswer):
        correct = abs(predicted.value - expected.value) < 1e-6
    elif isinstance(expected, StreetNameAnswer) and isinstance(predicted, StreetNameAnswer):
        correct = same_street_name(predicted.street_name, expected.street_name)
    elif isinstance(expected, StreetSetAnswer) and isinstance(predicted, StreetSetAnswer):
        f1 = set_f1(predicted.street_names, expected.street_names)
        return f1 == 1.0, f1, {"f1": f1}
    elif isinstance(expected, RouteAnswer) and isinstance(predicted, RouteAnswer):
        correct, sub_scores = score_route(view, linker, question, predicted)
        return correct, float(correct), sub_scores
    else:
        raise ValueError(f"Unexpected answer kind {expected.kind}")
    return correct, float(correct), {}
