"""Client for an OpenAI-compatible LLM endpoint (Ollama locally, vLLM on the server).

Every call is cached on disk by its full request, so a rerun of the same request
returns the stored response with its original token counts and latency. The
cache never changes what a run computes; it only avoids paying for a call twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from graph_llm.schemas.llm import LLMEndpointConfig

DEFAULT_CACHE_PATH = Path(".cache/llm_cache.sqlite")


@dataclass(frozen=True)
class ChatResult:
    """One chat completion with its cost."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool


class ResponseCache:
    """A key-value store of LLM responses in SQLite."""

    def __init__(self, path: Path) -> None:
        """Open or create the cache file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._connection.commit()

    def get(self, key: str) -> dict[str, object] | None:
        """Return a stored response, or None."""
        row = self._connection.execute(
            "SELECT value FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value: dict[str, object] = json.loads(row[0])
        return value

    def put(self, key: str, value: dict[str, object]) -> None:
        """Store a response."""
        self._connection.execute(
            "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self._connection.commit()


def request_key(payload: dict[str, object]) -> str:
    """Return the cache key of a request."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class LLMClient:
    """Chat and embedding calls against one configured endpoint."""

    def __init__(self, config: LLMEndpointConfig, cache_path: Path = DEFAULT_CACHE_PATH) -> None:
        """Create the client. The API key, if any, is read from the configured variable."""
        api_key = os.environ.get(config.api_key_env, "unused") if config.api_key_env else "unused"
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url, api_key=api_key, timeout=config.timeout_s, max_retries=2
        )
        self._cache = ResponseCache(cache_path)

    def chat(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> ChatResult:
        """Return a deterministic chat completion (temperature and seed from the config)."""
        payload: dict[str, object] = {
            "kind": "chat",
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        key = request_key(payload)
        stored = self._cache.get(key)
        if stored is not None:
            return ChatResult(
                text=str(stored["text"]),
                prompt_tokens=int(str(stored["prompt_tokens"])),
                completion_tokens=int(str(stored["completion_tokens"])),
                latency_s=float(str(stored["latency_s"])),
                cached=True,
            )
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.config.temperature,
            seed=self.config.seed,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        latency_s = time.perf_counter() - started
        usage = response.usage
        result = ChatResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=latency_s,
            cached=False,
        )
        self._cache.put(
            key,
            {
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_s": result.latency_s,
            },
        )
        return result

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text. Embeddings are not cached here."""
        response = self._client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]
