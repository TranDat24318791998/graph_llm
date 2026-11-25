"""Build the shared answering prompt and parse the LLM output into an answer."""

from __future__ import annotations

import json
import re

from graph_llm.schemas.prompt import AnswerPromptConfig
from graph_llm.schemas.question import (
    Answer,
    BooleanAnswer,
    NumberAnswer,
    Question,
    RouteAnswer,
    StreetNameAnswer,
    StreetSetAnswer,
)

JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
TRUE_WORDS = {"true", "có", "co", "yes", "đúng", "phải"}
FALSE_WORDS = {"false", "không", "khong", "no", "sai"}


def build_messages(
    prompt: AnswerPromptConfig, question: Question, context: str
) -> list[dict[str, str]]:
    """Return the chat messages for one question; only the context differs across methods."""
    user = prompt.user.format(
        context=context.strip() or prompt.empty_context,
        question=question.text,
        answer_format=prompt.answer_formats[question.answer.kind],
    )
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": user}]


def extract_json(text: str) -> dict[str, object] | None:
    """Return the first JSON object in a text, ignoring code fences and surrounding words."""
    match = JSON_OBJECT_PATTERN.search(text)
    if match is None:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def to_number(value: object) -> float | None:
    """Return a number from an int, float or numeric string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        found = re.search(r"-?\d+(?:[.,]\d+)?", value.replace(" ", ""))
        return float(found.group(0).replace(",", ".")) if found else None
    return None


def to_boolean(value: object) -> bool | None:
    """Return a boolean from a JSON boolean or a yes/no word."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().lower()
        if word in TRUE_WORDS:
            return True
        if word in FALSE_WORDS:
            return False
    return None


def parse_answer(kind: str, raw_response: str) -> Answer | None:
    """Parse the LLM output for an answer of the given kind; None if it does not fit."""
    data = extract_json(raw_response)
    if data is None:
        return None
    value = data.get("answer")
    if kind == "boolean":
        boolean = to_boolean(value)
        return BooleanAnswer(value=boolean) if boolean is not None else None
    if kind == "number":
        number = to_number(value)
        return NumberAnswer(value=number) if number is not None else None
    if kind == "street_name":
        return StreetNameAnswer(street_name=value.strip()) if isinstance(value, str) else None
    if kind == "street_set":
        if not isinstance(value, list):
            return None
        return StreetSetAnswer(street_names=[str(item).strip() for item in value])
    if kind == "route":
        streets = data.get("streets")
        length_m = to_number(data.get("length_m"))
        if not isinstance(streets, list) or length_m is None:
            return None
        return RouteAnswer(street_names=[str(item).strip() for item in streets], length_m=length_m)
    raise ValueError(f"Unknown answer kind: {kind}")
