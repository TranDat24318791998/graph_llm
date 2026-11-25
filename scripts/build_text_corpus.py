"""Generate the street description corpus of a processed graph and embed its chunks.

Writes data/processed/corpus/<corpus_id>/.

Usage:
    uv run python scripts/build_text_corpus.py hcmc_district1_v2_p1 \
        configs/llm/ollama_qwen2.5_7b.yaml
"""

import argparse
from pathlib import Path

import yaml

from graph_llm.llm.client import LLMClient
from graph_llm.questions.graph_view import load_graph_view
from graph_llm.representations.corpus_store import embed_texts, save_corpus
from graph_llm.representations.text_corpus import build_corpus
from graph_llm.schemas.llm import LLMEndpointConfig

PROCESSED_DATA_DIRECTORY = Path("data/processed")
RAW_DATA_DIRECTORY = Path("data/raw")
EMBEDDING_MODEL = "bge-m3"


def main() -> None:
    """Parse arguments, generate the corpus, embed it and write it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_version")
    parser.add_argument("llm_config", type=Path, help="Endpoint that also serves embeddings.")
    parser.add_argument(
        "--text-version", default="v1", help="Version of the text templates, part of the id."
    )
    arguments = parser.parse_args()

    view = load_graph_view(PROCESSED_DATA_DIRECTORY, RAW_DATA_DIRECTORY, arguments.graph_version)
    documents, chunks = build_corpus(view)
    llm_config = LLMEndpointConfig.model_validate(
        yaml.safe_load(arguments.llm_config.read_text(encoding="utf-8"))
    )
    embeddings = embed_texts(LLMClient(llm_config), EMBEDDING_MODEL, [c.text for c in chunks])
    corpus_id = f"{arguments.graph_version}_text_{arguments.text_version}"
    manifest = save_corpus(
        corpus_id, arguments.graph_version, documents, chunks, embeddings, EMBEDDING_MODEL
    )
    counts = f"{manifest.document_count} documents, {manifest.chunk_count} chunks"
    print(f"Wrote corpus {corpus_id}: {counts}")


if __name__ == "__main__":
    main()
