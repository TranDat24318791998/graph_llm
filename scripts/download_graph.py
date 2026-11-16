"""Download and freeze the OSM road graph of a study area.

Usage:
    uv run python scripts/download_graph.py configs/areas/hcmc_district1.yaml
"""

import argparse
from pathlib import Path

from graph_llm.graph.snapshot import download_snapshot, load_study_area

RAW_DATA_DIRECTORY = Path("data/raw")


def main() -> None:
    """Parse arguments and download the snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("area_config", type=Path, help="Path to a study area YAML file.")
    arguments = parser.parse_args()

    area = load_study_area(arguments.area_config)
    metadata = download_snapshot(area, RAW_DATA_DIRECTORY)
    print(f"Downloaded {metadata.area.area_id}:")
    print(f"  nodes: {metadata.node_count}")
    print(f"  edges: {metadata.edge_count}")
    print(f"  sha256: {metadata.graphml_sha256}")


if __name__ == "__main__":
    main()
