"""Generate, paraphrase and freeze a benchmark version.

Writes data/benchmark/<version>/questions.jsonl and manifest.json.

Usage:
    uv run python scripts/build_benchmark.py configs/benchmark/v0_1.yaml \
        configs/llm/ollama_qwen2.5_7b.yaml
"""

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

from graph_llm.graph.snapshot import compute_sha256
from graph_llm.llm.client import LLMClient
from graph_llm.questions.generate import generate_questions
from graph_llm.questions.graph_view import load_graph_view
from graph_llm.questions.paraphrase import paraphrase_question
from graph_llm.schemas.benchmark import BenchmarkConfig, BenchmarkManifest
from graph_llm.schemas.llm import LLMEndpointConfig

PROCESSED_DATA_DIRECTORY = Path("data/processed")
RAW_DATA_DIRECTORY = Path("data/raw")
BENCHMARK_DIRECTORY = Path("data/benchmark")


def main() -> None:
    """Parse arguments, build the benchmark and write it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_config", type=Path)
    parser.add_argument("llm_config", type=Path, help="LLM used for paraphrasing.")
    arguments = parser.parse_args()

    config = BenchmarkConfig.model_validate(
        yaml.safe_load(arguments.benchmark_config.read_text(encoding="utf-8"))
    )
    llm_config = LLMEndpointConfig.model_validate(
        yaml.safe_load(arguments.llm_config.read_text(encoding="utf-8"))
    )
    target_directory = BENCHMARK_DIRECTORY / config.benchmark_version
    if target_directory.exists():
        raise FileExistsError(f"{target_directory} exists; benchmark versions are immutable.")

    view = load_graph_view(PROCESSED_DATA_DIRECTORY, RAW_DATA_DIRECTORY, config.graph_version)
    templates = {template.template_id: template for template in config.templates}
    client = LLMClient(llm_config)
    questions = []
    fallback_ids = []
    for question in generate_questions(view, config):
        paraphrased, kept = paraphrase_question(
            client, config.paraphrase_instruction, question, templates[question.template_id]
        )
        questions.append(paraphrased)
        if not kept:
            fallback_ids.append(question.question_id)

    target_directory.mkdir(parents=True)
    questions_path = target_directory / "questions.jsonl"
    questions_path.write_text(
        "".join(question.model_dump_json() + "\n" for question in questions), encoding="utf-8"
    )
    manifest = BenchmarkManifest(
        config=config,
        created_at=datetime.now(UTC),
        question_count=len(questions),
        counts_by_template=dict(Counter(question.template_id for question in questions)),
        paraphrase_model=llm_config.model,
        paraphrase_fallback_question_ids=fallback_ids,
        questions_sha256=compute_sha256(questions_path),
    )
    (target_directory / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(questions)} questions to {questions_path}")
    kept_count = len(questions) - len(fallback_ids)
    print(f"Paraphrase kept for {kept_count}, fallback for {len(fallback_ids)}")


if __name__ == "__main__":
    main()
