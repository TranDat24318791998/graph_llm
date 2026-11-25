"""Schemas for the text corpus generated from the graph (input of M1 and M2)."""

from datetime import datetime

from graph_llm.schemas.base import StrictModel
from graph_llm.schemas.question import EdgeRef


class TextDocument(StrictModel):
    """The description of one street."""

    document_id: str
    street_id: str
    text: str


class TextChunk(StrictModel):
    """A retrievable piece of a document, with the graph edges it describes."""

    chunk_id: str
    document_id: str
    text: str
    edges: list[EdgeRef]


class CorpusManifest(StrictModel):
    """Provenance of a generated corpus."""

    corpus_id: str
    graph_version: str
    created_at: datetime
    document_count: int
    chunk_count: int
    character_count: int
    embedding_model: str
    documents_sha256: str
    chunks_sha256: str
    embeddings_sha256: str
