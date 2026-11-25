"""Build the LightRAG (M2) index over a text corpus and record its cost.

Writes data/processed/lightrag/<index_id>/ (LightRAG files) and index_manifest.json.

Usage:
    uv run python scripts/build_lightrag_index.py hcmc_district1_v2_p1_text_v1 \
        configs/llm/ollama_qwen2.5_14b.yaml
"""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from graph_llm.representations.corpus_store import load_corpus
from graph_llm.representations.lightrag_m2 import CostLog, build_index, pack_documents
from graph_llm.schemas.llm import LLMEndpointConfig

LIGHTRAG_DIRECTORY = Path("data/processed/lightrag")
EMBEDDING_MODEL = "bge-m3"


def main() -> None:
    """Parse arguments, build the index and write its manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_id")
    parser.add_argument("llm_config", type=Path, help="LLM used for entity extraction.")
    arguments = parser.parse_args()

    llm_config = LLMEndpointConfig.model_validate(
        yaml.safe_load(arguments.llm_config.read_text(encoding="utf-8"))
    )
    corpus = load_corpus(arguments.corpus_id)
    index_id = f"{arguments.corpus_id}_packed_{llm_config.model}"
    working_directory = LIGHTRAG_DIRECTORY / index_id
    if (working_directory / "index_manifest.json").exists():
        raise FileExistsError(f"{working_directory} is complete; indexes are immutable.")
    working_directory.mkdir(parents=True, exist_ok=True)

    pack_ids, pack_texts = pack_documents([document.text for document in corpus.documents])
    cost_log = CostLog()
    started = time.perf_counter()
    asyncio.run(
        build_index(
            working_directory,
            llm_config,
            EMBEDDING_MODEL,
            pack_ids,
            pack_texts,
            cost_log,
        )
    )
    manifest = {
        "index_id": index_id,
        "corpus_id": arguments.corpus_id,
        "llm_config": llm_config.model_dump(),
        "embedding_model": EMBEDDING_MODEL,
        "entity_extract_max_gleaning": 0,
        "document_packs": len(pack_ids),
        "created_at": datetime.now(UTC).isoformat(),
        "wall_clock_seconds": time.perf_counter() - started,
        "llm_calls": cost_log.calls,
        "llm_prompt_tokens": cost_log.prompt_tokens,
        "llm_completion_tokens": cost_log.completion_tokens,
        "llm_seconds": cost_log.seconds,
    }
    (working_directory / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
