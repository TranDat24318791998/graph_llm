import pytest

from graph_llm.evaluation.answering import parse_answer
from graph_llm.evaluation.scoring import score_answer, set_f1
from graph_llm.questions.graph_view import GraphView
from graph_llm.representations.linker import StreetLinker
from graph_llm.representations.subgraph import build_subgraph_context
from graph_llm.schemas.question import (
    BooleanAnswer,
    Evidence,
    NumberAnswer,
    Question,
    QuestionFeatures,
    QuestionType,
    RouteAnswer,
    StreetSetAnswer,
)


def route_question() -> Question:
    """From node 1 to node 7. Shortest legal route: Pasteur 1-2-3, Hai Ba Trung 3-7, 300 m."""
    return Question(
        question_id="L4_test",
        benchmark_version="test",
        question_type=QuestionType.L4_ROUTE,
        template_id="L4_shortest_route",
        template_text="t",
        text="t",
        slots={"start_node_ids": "1", "end_node_ids": "7"},
        answer=RouteAnswer(
            street_names=["Pasteur", "Hai Bà Trưng"], length_m=300.0, node_path=[1, 2, 3, 7]
        ),
        evidence=Evidence(),
        features=QuestionFeatures(
            hop_count=3, requires_computation=True, depends_on_edge_direction=True
        ),
    )


def test_parse_answers_in_several_shapes() -> None:
    assert parse_answer("boolean", '```json\n{"answer": true}\n```') == BooleanAnswer(value=True)
    assert parse_answer("boolean", '{"answer": "Có"}') == BooleanAnswer(value=True)
    assert parse_answer("number", '{"answer": "3 đường"}') == NumberAnswer(value=3.0)
    assert parse_answer("street_set", '{"answer": ["A", "B"]}') == StreetSetAnswer(
        street_names=["A", "B"]
    )
    route = parse_answer("route", 'Đáp án: {"streets": ["X", "Y"], "length_m": 1200}')
    assert route == RouteAnswer(street_names=["X", "Y"], length_m=1200.0)
    assert parse_answer("number", "không biết") is None
    assert parse_answer("boolean", '{"answer": "có lẽ"}') is None


def test_set_f1() -> None:
    assert set_f1(["Lê Lợi", "Pasteur"], ["Pasteur", "Lê Lợi"]) == 1.0
    assert set_f1(["Đường Lê Lợi", "le loi"], ["Lê Lợi"]) == 1.0
    assert set_f1(["Lê Lợi", "Hàm Nghi"], ["Lê Lợi", "Pasteur"]) == pytest.approx(0.5)
    assert set_f1([], ["Lê Lợi"]) == 0.0


def test_linker_finds_names_and_resolves_without_diacritics(view: GraphView) -> None:
    linker = StreetLinker.from_view(view)

    assert linker.link("Từ giao lộ Hai Bà Trưng và đường Pasteur đi đâu?") == {
        "hai-ba-trung",
        "pasteur",
    }
    assert linker.link("đường 10A") == {"10a"}
    assert linker.resolve("Le Loi") == {"le-loi"}
    assert linker.resolve("Hàm Nghi") == set()


def test_correct_route_is_scored_correct(view: GraphView) -> None:
    linker = StreetLinker.from_view(view)
    predicted = RouteAnswer(street_names=["Pasteur", "Hai Bà Trưng"], length_m=310.0)

    correct, score, sub_scores = score_answer(view, linker, route_question(), predicted)

    assert correct and score == 1.0
    assert sub_scores["legal"] == 1.0
    assert sub_scores["length_within_10_percent"] == 1.0
    assert sub_scores["sequence_exact"] == 1.0


def test_route_against_one_way_direction_is_flagged(view: GraphView) -> None:
    linker = StreetLinker.from_view(view)
    question = route_question().model_copy(
        update={"slots": {"start_node_ids": "3", "end_node_ids": "8"}}
    )
    predicted = RouteAnswer(street_names=["Pasteur", "Nguyễn Huệ"], length_m=300.0)

    correct, _, sub_scores = score_answer(view, linker, question, predicted)

    assert not correct
    assert sub_scores["legal"] == 0.0
    assert sub_scores["breaks_direction_or_turn_rule"] == 1.0


def test_route_with_unknown_street_is_wrong(view: GraphView) -> None:
    linker = StreetLinker.from_view(view)
    predicted = RouteAnswer(street_names=["Pasteur", "Hàm Nghi"], length_m=300.0)

    correct, _, sub_scores = score_answer(view, linker, route_question(), predicted)

    assert not correct
    assert sub_scores["all_streets_known"] == 0.0


def test_subgraph_marks_one_way_and_two_way_segments(view: GraphView) -> None:
    text, covered = build_subgraph_context(view, {"pasteur"}, hops=0, budget_characters=10_000)

    lines = text.splitlines()[1:]
    assert sorted(lines) == ["n1 -> n2 | Pasteur | 100 m", "n2 -> n3 | Pasteur | 100 m"]
    assert len(covered) == 2


def test_subgraph_respects_the_budget(view: GraphView) -> None:
    full, _ = build_subgraph_context(view, {"pasteur"}, hops=2, budget_characters=100_000)
    short, _ = build_subgraph_context(view, {"pasteur"}, hops=2, budget_characters=len(full) // 2)

    assert len(short) <= len(full) // 2
    assert "<->" in full


def test_wilson_interval_and_mcnemar() -> None:
    from graph_llm.analysis.statistics import mcnemar_exact_p_value, wilson_interval

    low, high = wilson_interval(25, 50)
    assert low == pytest.approx(0.366, abs=0.001)
    assert high == pytest.approx(0.634, abs=0.001)
    assert mcnemar_exact_p_value(0, 0) == 1.0
    assert mcnemar_exact_p_value(10, 0) == pytest.approx(2 / 1024)


def test_subgraph_can_name_nodes_by_their_intersecting_streets(view: GraphView) -> None:
    from graph_llm.representations.subgraph import NodeNaming

    text, _ = build_subgraph_context(
        view, {"pasteur"}, hops=0, budget_characters=10_000, naming=NodeNaming.INTERSECTION_NAMES
    )

    lines = text.splitlines()[1:]
    assert sorted(lines) == [
        "[Lê Lợi / Pasteur] -> [Hai Bà Trưng / Pasteur] | Pasteur | 100 m",
        "[Nguyễn Huệ / Pasteur] -> [Lê Lợi / Pasteur] | Pasteur | 100 m",
    ]


def test_graph_linked_text_orders_streets_by_street_hops(view: GraphView) -> None:
    from graph_llm.representations.graph_linked_text import streets_in_hop_order

    names = [view.streets[s].display_name for s in streets_in_hop_order(view, {"pasteur"}, hops=2)]

    assert names[0] == "Pasteur"
    assert set(names[1:4]) == {"Hai Bà Trưng", "Lê Lợi", "Nguyễn Huệ"}
    assert names[4:] == ["Đường 10A"]


def test_linker_can_be_restricted_to_street_categories() -> None:
    from conftest import CONFIG, build_hand_checked_view
    from graph_llm.graph.streets import build_street_layer
    from graph_llm.schemas.street import StreetCategory

    graph = build_hand_checked_view().graph
    config = CONFIG.model_copy(update={"internal_road_names": ["Nguyễn Huệ"]})
    view = GraphView.from_graph_and_streets(graph, build_street_layer(graph, config).streets)
    text = "Từ Pasteur đi Nguyễn Huệ, không tính đường nội bộ."

    everything = StreetLinker.from_view(view)
    regular_only = StreetLinker.from_view(view, frozenset({StreetCategory.REGULAR}))

    assert len(everything.link(text)) == 2
    assert regular_only.link(text) == {"pasteur"}
