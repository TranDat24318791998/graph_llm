"""Schema of an LLM endpoint configuration."""

from pydantic import Field

from graph_llm.schemas.base import StrictModel


class LLMEndpointConfig(StrictModel):
    """Where and how to call an OpenAI-compatible LLM endpoint."""

    base_url: str = Field(description="For example http://127.0.0.1:11434/v1 for Ollama.")
    model: str
    api_key_env: str | None = Field(
        default=None, description="Environment variable holding the API key; None if not needed."
    )
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 1024
    timeout_s: float = 600.0
