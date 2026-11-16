import pytest
from pydantic import ValidationError

from graph_llm.schemas import (
    BooleanAnswer,
    EdgeRef,
    Evidence,
    Question,
    QuestionFeatures,
    QuestionType,
    RouteAnswer,
    Score,
)
from graph_llm.schemas.run import ErrorStage


def make_question() -> Question:
    return Question(
        question_id="L1-000001",
        benchmark_version="v0.1",
        question_type=QuestionType.L1_ATTRIBUTE,
        template_id="l1_oneway",
        template_text="Đường Pasteur có phải đường một chiều không?",
        text="Pasteur có phải là đường một chiều không?",
        slots={"street": "Pasteur"},
        answer=BooleanAnswer(value=True),
        evidence=Evidence(edges=[EdgeRef(source_node_id=1, target_node_id=2)]),
        features=QuestionFeatures(
            hop_count=0,
            requires_computation=False,
            depends_on_edge_direction=False,
        ),
    )


def test_question_round_trips_through_json() -> None:
    question = make_question()

    restored = Question.model_validate_json(question.model_dump_json())

    assert restored == question


def test_answer_kind_selects_the_answer_model() -> None:
    payload = make_question().model_dump()
    payload["answer"] = {"kind": "route", "street_names": ["Lê Lợi", "Pasteur"], "length_m": 420.5}

    question = Question.model_validate(payload)

    assert isinstance(question.answer, RouteAnswer)


def test_unknown_field_is_rejected() -> None:
    payload = make_question().model_dump()
    payload["unexpected_field"] = "value"

    with pytest.raises(ValidationError):
        Question.model_validate(payload)


def test_records_are_immutable() -> None:
    question = make_question()

    with pytest.raises(ValidationError):
        question.text = "changed"  # type: ignore[misc]


def test_score_outside_unit_interval_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Score(
            run_id="run-1",
            question_id="L1-000001",
            is_correct=True,
            score=1.5,
            error_stage=ErrorStage.NONE,
        )
