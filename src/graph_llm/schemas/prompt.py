"""Schema of the answering prompt."""

from graph_llm.schemas.base import StrictModel


class AnswerPromptConfig(StrictModel):
    """The answering prompt shared by all representations."""

    prompt_version: str
    system: str
    user: str
    empty_context: str
    answer_formats: dict[str, str]
