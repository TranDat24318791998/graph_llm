"""M1: dense retrieval of corpus chunks by cosine similarity of bge-m3 embeddings."""

from __future__ import annotations

import numpy as np

from graph_llm.llm.client import LLMClient
from graph_llm.representations.corpus_store import Corpus, embed_texts
from graph_llm.schemas.question import EdgeRef

CHUNK_SEPARATOR = "\n\n"


def retrieve_chunks(
    client: LLMClient,
    corpus: Corpus,
    question_text: str,
    top_k: int,
    budget_characters: int,
) -> tuple[str, set[EdgeRef]]:
    """Return the best chunks that fit the budget, joined, and the edges they describe."""
    query = embed_texts(client, corpus.manifest.embedding_model, [question_text])[0]
    similarities = corpus.embeddings @ query
    ranking = np.argsort(-similarities, kind="stable")[:top_k]
    texts: list[str] = []
    used = 0
    covered: set[EdgeRef] = set()
    for index in ranking:
        chunk = corpus.chunks[int(index)]
        cost = len(chunk.text) + len(CHUNK_SEPARATOR)
        if used + cost > budget_characters:
            continue
        texts.append(chunk.text)
        used += cost
        covered.update(chunk.edges)
    return CHUNK_SEPARATOR.join(texts), covered
