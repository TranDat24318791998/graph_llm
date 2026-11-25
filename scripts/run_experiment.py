"""Run representations of an experiment and write one run directory per representation.

Usage:
    uv run python scripts/run_experiment.py configs/experiments/exp001.yaml M1 M3
"""

import argparse
from pathlib import Path

import yaml

from graph_llm.evaluation.runner import run_representation
from graph_llm.schemas.experiment import ExperimentConfig
from graph_llm.schemas.run import Representation


def main() -> None:
    """Parse arguments and run each requested representation in turn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("representations", nargs="+", choices=[r.value for r in Representation])
    arguments = parser.parse_args()

    config = ExperimentConfig.model_validate(
        yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    )
    for name in arguments.representations:
        representation = Representation(name)
        if representation not in config.representations:
            raise ValueError(f"{name} is not listed in {arguments.config}")
        metrics = run_representation(config, representation)
        print(f"{metrics.run_id}: accuracy {metrics.overall.accuracy:.3f}")
        for question_type, group in metrics.by_question_type.items():
            print(f"  {question_type}: accuracy {group.accuracy:.3f}, score {group.mean_score:.3f}")


if __name__ == "__main__":
    main()
