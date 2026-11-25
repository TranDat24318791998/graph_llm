"""Store and load the text corpus with its chunk embeddings, frozen with checksums."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from graph_llm.graph.snapshot import compute_sha256
from graph_llm.llm.client import LLMClient
from graph_llm.schemas.corpus import CorpusManifest, TextChunk, TextDocument

CORPUS_DIRECTORY = Path("data/processed/corpus")
EMBEDDING_BATCH_SIZE = 32


@dataclass(frozen=True)
class Corpus:
    """A loaded corpus: documents, chunks and one normalized embedding per chunk."""

    manifest: CorpusManifest
    documents: list[TextDocument]
    chunks: list[TextChunk]
    embeddings: npt.NDArray[np.float32]


def normalize_rows(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Scale every row to unit length, so a dot product is the cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized: npt.NDArray[np.float32] = (matrix / np.maximum(norms, 1e-12)).astype(np.float32)
    return normalized


def embed_texts(client: LLMClient, model: str, texts: list[str]) -> npt.NDArray[np.float32]:
    """Embed texts in batches and return normalized rows."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors += client.embed(model, texts[start : start + EMBEDDING_BATCH_SIZE])
    return normalize_rows(np.array(vectors, dtype=np.float32))


def save_corpus(
    corpus_id: str,
    graph_version: str,
    documents: list[TextDocument],
    chunks: list[TextChunk],
    embeddings: npt.NDArray[np.float32],
    embedding_model: str,
    directory: Path = CORPUS_DIRECTORY,
) -> CorpusManifest:
    """Write a corpus once.

    Raises:
        FileExistsError: If the corpus id already exists.
    """
    target = directory / corpus_id
    if target.exists():
        raise FileExistsError(f"{target} exists; corpora are immutable.")
    target.mkdir(parents=True)
    documents_path = target / "documents.jsonl"
    chunks_path = target / "chunks.jsonl"
    embeddings_path = target / "embeddings.npy"
    documents_path.write_text(
        "".join(document.model_dump_json() + "\n" for document in documents), encoding="utf-8"
    )
    chunks_path.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8"
    )
    np.save(embeddings_path, embeddings)
    manifest = CorpusManifest(
        corpus_id=corpus_id,
        graph_version=graph_version,
        created_at=datetime.now(UTC),
        document_count=len(documents),
        chunk_count=len(chunks),
        character_count=sum(len(document.text) for document in documents),
        embedding_model=embedding_model,
        documents_sha256=compute_sha256(documents_path),
        chunks_sha256=compute_sha256(chunks_path),
        embeddings_sha256=compute_sha256(embeddings_path),
    )
    (target / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def load_corpus(corpus_id: str, directory: Path = CORPUS_DIRECTORY) -> Corpus:
    """Load a corpus and verify its checksums.

    Raises:
        ValueError: If a file does not match its recorded checksum.
    """
    source = directory / corpus_id
    manifest = CorpusManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in (
        ("documents.jsonl", manifest.documents_sha256),
        ("chunks.jsonl", manifest.chunks_sha256),
        ("embeddings.npy", manifest.embeddings_sha256),
    ):
        if compute_sha256(source / name) != expected:
            raise ValueError(f"Checksum mismatch for {source / name}.")
    documents = [
        TextDocument.model_validate_json(line)
        for line in (source / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    chunks = [
        TextChunk.model_validate_json(line)
        for line in (source / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    embeddings: npt.NDArray[np.float32] = np.load(source / "embeddings.npy")
    return Corpus(manifest=manifest, documents=documents, chunks=chunks, embeddings=embeddings)
