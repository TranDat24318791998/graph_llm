"""Generate the result tables of an experiment from its run directories.

Writes <results_dir>/results.md and <results_dir>/per_question.csv.

Usage:
    uv run python scripts/analyze_experiment.py configs/experiments/exp001.yaml \
        experiments/EXP-001-phase1-pilot/results runs/<run_id> [runs/<run_id> ...]
"""

import argparse
import json
from pathlib import Path

import yaml

from graph_llm.analysis.experiment_report import (
    load_run,
    render_report,
    with_unique_names,
    write_per_question_table,
)
from graph_llm.analysis.lightrag_alignment import align_lightrag_graph
from graph_llm.evaluation.runner import (
    LIGHTRAG_DIRECTORY,
    PROCESSED_DATA_DIRECTORY,
    RAW_DATA_DIRECTORY,
    load_questions,
)
from graph_llm.questions.graph_view import load_graph_view
from graph_llm.schemas.experiment import ExperimentConfig
from graph_llm.schemas.run import Representation


def main() -> None:
    """Parse arguments, load runs and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("run_directories", type=Path, nargs="+")
    arguments = parser.parse_args()

    config = ExperimentConfig.model_validate(
        yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    )
    questions = load_questions(config.benchmark_version)
    runs = with_unique_names(
        sorted(
            (load_run(path) for path in arguments.run_directories),
            key=lambda r: (r.manifest.experiment_id, r.manifest.representation.value),
        )
    )
    has_m2_run = any(
        run.manifest.representation is Representation.M2_LLM_BUILT_GRAPH for run in runs
    )
    index_directory = LIGHTRAG_DIRECTORY / config.lightrag_index_id
    alignment = None
    index_manifest = None
    if has_m2_run and (index_directory / "index_manifest.json").exists():
        view = load_graph_view(PROCESSED_DATA_DIRECTORY, RAW_DATA_DIRECTORY, config.graph_version)
        alignment = align_lightrag_graph(index_directory, view)
        index_manifest = json.loads((index_directory / "index_manifest.json").read_text())

    arguments.results_directory.mkdir(parents=True, exist_ok=True)
    report = render_report(config.experiment_id, questions, runs, alignment, index_manifest)
    (arguments.results_directory / "results.md").write_text(report, encoding="utf-8")
    write_per_question_table(arguments.results_directory / "per_question.csv", questions, runs)
    print(f"Wrote {arguments.results_directory / 'results.md'}")


if __name__ == "__main__":
    main()
