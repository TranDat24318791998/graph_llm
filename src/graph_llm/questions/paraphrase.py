"""Paraphrase template questions with an LLM, keeping only paraphrases that pass a check.

The check is automatic: every street name of the question must appear unchanged,
and every required phrase of the template must still be present. A question
whose paraphrase fails keeps its template text, and its id is recorded.
"""

import unicodedata

from graph_llm.llm.client import LLMClient
from graph_llm.schemas.benchmark import QuestionTemplate
from graph_llm.schemas.question import Question

NODE_SLOT_SUFFIX = "_node_ids"


def normalized(text: str) -> str:
    """Return text in NFC and lower case, for containment checks."""
    return unicodedata.normalize("NFC", text).lower()


def paraphrase_is_valid(question: Question, template: QuestionTemplate, paraphrase: str) -> bool:
    """Return True if a paraphrase keeps every street name and required phrase."""
    text = normalized(paraphrase)
    if not text.strip() or "\n" in paraphrase.strip():
        return False
    names = [value for slot, value in question.slots.items() if not slot.endswith(NODE_SLOT_SUFFIX)]
    phrases = names + template.required_phrases
    return all(normalized(phrase) in text for phrase in phrases)


def paraphrase_question(
    client: LLMClient, instruction: str, question: Question, template: QuestionTemplate
) -> tuple[Question, bool]:
    """Return the question with its paraphrased text, and whether the paraphrase was kept."""
    result = client.chat(
        [{"role": "user", "content": f"{instruction}\nCâu hỏi: {question.template_text}"}],
        max_tokens=300,
    )
    paraphrase = result.text.strip().strip('"').strip()
    if paraphrase_is_valid(question, template, paraphrase):
        return question.model_copy(update={"text": paraphrase}), True
    return question, False
