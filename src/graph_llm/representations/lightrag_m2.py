"""M2: LightRAG over the generated street descriptions.

LightRAG builds its own knowledge graph from the text with an LLM; the OSM graph
is never shown to it. At query time LightRAG only builds the context (entities,
relations and source chunks it selects); the answer is produced by the shared
answering prompt, like every other representation.

Facts about LightRAG 1.5.8 that matter here (verified in the installed source):
its graph store is an undirected ``networkx.Graph`` and its extraction prompt
asks for undirected relations, so one-way direction can survive only inside
description text. Keyword extraction at query time calls the LLM once.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI

from graph_llm.schemas.llm import LLMEndpointConfig

PACK_CHARACTER_LIMIT = 3000
EMBEDDING_DIMENSION = 1024
EMBEDDING_MAX_TOKENS = 8192
LANGUAGE = "Vietnamese"


def pack_documents(
    texts: list[str], limit: int = PACK_CHARACTER_LIMIT
) -> tuple[list[str], list[str]]:
    """Join consecutive short documents into packs of at most ``limit`` characters.

    Every LightRAG extraction call carries about 6,000 tokens of fixed prompt, and
    LightRAG chunks each document on its own, so 520 mostly short street documents
    would cost over 500 calls. Packs of about one LightRAG chunk (1,200 tokens, about
    3,000 characters of this Vietnamese text) keep the text unchanged and cut the
    number of calls about fourfold. A document longer than the limit is its own pack.

    Returns:
        Pack ids and pack texts.
    """
    packs: list[list[str]] = []
    pack_length = 0
    for text in texts:
        if packs and pack_length + len(text) + 2 <= limit:
            packs[-1].append(text)
            pack_length += len(text) + 2
        else:
            packs.append([text])
            pack_length = len(text)
    pack_ids = [f"pack-{index:04d}" for index in range(len(packs))]
    return pack_ids, ["\n\n".join(pack) for pack in packs]


@dataclass
class CostLog:
    """Running totals of LLM calls made through LightRAG."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)

    def snapshot(self) -> tuple[int, int, int, float]:
        """Return the current totals."""
        return self.calls, self.prompt_tokens, self.completion_tokens, self.seconds


def make_llm_function(config: LLMEndpointConfig, cost_log: CostLog) -> Any:
    """Return an async LLM function with the signature LightRAG expects."""
    client = AsyncOpenAI(base_url=config.base_url, api_key="unused", timeout=config.timeout_s)

    async def llm_function(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        kwargs.pop("hashing_kv", None)
        kwargs.pop("enable_cot", None)
        kwargs.pop("stream", None)
        response_format = kwargs.pop("response_format", None)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages += history_messages or []
        messages.append({"role": "user", "content": prompt})
        extra: dict[str, Any] = {"response_format": response_format} if response_format else {}
        started = time.perf_counter()
        response = await client.chat.completions.create(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=config.temperature,
            seed=config.seed,
            max_tokens=config.max_tokens,
            **extra,
        )
        cost_log.calls += 1
        cost_log.seconds += time.perf_counter() - started
        if response.usage is not None:
            cost_log.prompt_tokens += response.usage.prompt_tokens
            cost_log.completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content or ""

    return llm_function


def make_embedding_function(config: LLMEndpointConfig, model: str) -> EmbeddingFunc:
    """Return the bge-m3 embedding function, served by the same endpoint."""
    client = AsyncOpenAI(base_url=config.base_url, api_key="unused", timeout=config.timeout_s)

    async def embed(texts: list[str], **kwargs: Any) -> npt.NDArray[np.float32]:
        response = await client.embeddings.create(model=model, input=texts)
        return np.array([item.embedding for item in response.data], dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIMENSION,
        max_token_size=EMBEDDING_MAX_TOKENS,
        func=embed,
        model_name=model,
    )


def create_lightrag(
    working_directory: Path,
    llm_config: LLMEndpointConfig,
    embedding_model: str,
    cost_log: CostLog,
    use_cache: bool,
) -> LightRAG:
    """Create a LightRAG instance on a working directory."""
    return LightRAG(
        working_dir=str(working_directory),
        llm_model_func=make_llm_function(llm_config, cost_log),
        llm_model_name=llm_config.model,
        embedding_func=make_embedding_function(llm_config, embedding_model),
        addon_params={"language": LANGUAGE},
        entity_extract_max_gleaning=0,
        llm_model_max_async=4,
        default_llm_timeout=int(llm_config.timeout_s),
        enable_llm_cache=use_cache,
        enable_llm_cache_for_entity_extract=use_cache,
    )


async def build_index(
    working_directory: Path,
    llm_config: LLMEndpointConfig,
    embedding_model: str,
    document_ids: list[str],
    texts: list[str],
    cost_log: CostLog,
) -> None:
    """Insert all documents into a new LightRAG working directory."""
    rag = create_lightrag(working_directory, llm_config, embedding_model, cost_log, use_cache=True)
    await rag.initialize_storages()
    try:
        await rag.ainsert(
            texts, ids=document_ids, file_paths=[f"{doc_id}.txt" for doc_id in document_ids]
        )
    finally:
        await rag.finalize_storages()


@dataclass(frozen=True)
class LightRAGQuerySettings:
    """Retrieval settings of M2."""

    mode: str
    top_k: int
    chunk_top_k: int
    max_entity_tokens: int
    max_relation_tokens: int
    max_total_tokens: int


class LightRAGRetriever:
    """Builds M2 contexts with LightRAG, reusing one event loop and one instance."""

    def __init__(
        self,
        working_directory: Path,
        llm_config: LLMEndpointConfig,
        embedding_model: str,
        settings: LightRAGQuerySettings,
    ) -> None:
        """Open the index; query-time LLM calls (keyword extraction) use ``llm_config``."""
        self.cost_log = CostLog()
        self.settings = settings
        self._loop = asyncio.new_event_loop()
        self._rag = create_lightrag(
            working_directory, llm_config, embedding_model, self.cost_log, use_cache=False
        )
        self._loop.run_until_complete(self._rag.initialize_storages())

    def context(self, question_text: str) -> str:
        """Return LightRAG's context for a question, or an empty string if it finds nothing."""
        param = QueryParam(
            mode=self.settings.mode,
            only_need_context=True,
            enable_rerank=False,
            top_k=self.settings.top_k,
            chunk_top_k=self.settings.chunk_top_k,
            max_entity_tokens=self.settings.max_entity_tokens,
            max_relation_tokens=self.settings.max_relation_tokens,
            max_total_tokens=self.settings.max_total_tokens,
        )
        result = self._loop.run_until_complete(self._rag.aquery_llm(question_text, param=param))
        if result.get("status") != "success":
            return ""
        return str(result["llm_response"]["content"])

    def close(self) -> None:
        """Flush and close the index."""
        self._loop.run_until_complete(self._rag.finalize_storages())
        self._loop.close()
