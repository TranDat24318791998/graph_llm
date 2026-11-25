"""Build and freeze the processed graph described by a graph processing config.

Writes data/processed/<graph_version>/ and docs/reports/street_model_<graph_version>.md.

Usage:
    uv run python scripts/build_graph.py configs/graphs/hcmc_district1_v2_p1.yaml
"""

import argparse
from pathlib import Path

from graph_llm.graph.processed import (
    load_processing_config,
    process_snapshot,
    save_processed_graph,
)
from graph_llm.graph.snapshot import load_snapshot
from graph_llm.graph.street_report import render_street_model_markdown

RAW_DATA_DIRECTORY = Path("data/raw")
PROCESSED_DATA_DIRECTORY = Path("data/processed")
REPORT_DIRECTORY = Path("docs/reports")


def main() -> None:
    """Parse arguments, process the snapshot and write the graph and its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a graph processing YAML file.")
    arguments = parser.parse_args()

    config = load_processing_config(arguments.config)
    snapshot_graph, snapshot_metadata = load_snapshot(RAW_DATA_DIRECTORY, config.source_area_id)
    result = process_snapshot(snapshot_graph, config)
    metadata = save_processed_graph(result, config, snapshot_metadata, PROCESSED_DATA_DIRECTORY)

    report_path = REPORT_DIRECTORY / f"street_model_{config.graph_version}.md"
    report_path.write_text(
        render_street_model_markdown(metadata, result.street_layer.streets), encoding="utf-8"
    )
    print(f"Built {config.graph_version}:")
    print(f"  nodes: {metadata.node_count}, edges: {metadata.edge_count}")
    print(f"  streets: {metadata.street_count}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
