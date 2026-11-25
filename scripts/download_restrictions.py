"""Download and freeze the OSM turn restrictions of a study area.

Writes data/raw/<area_id>_restrictions/. Like a snapshot, this is done once per
area id and never repeated.

Usage:
    uv run python scripts/download_restrictions.py configs/areas/hcmc_district1_v2.yaml
"""

import argparse
from pathlib import Path

from graph_llm.graph.restrictions import download_restrictions
from graph_llm.graph.snapshot import load_study_area

RAW_DATA_DIRECTORY = Path("data/raw")


def main() -> None:
    """Parse arguments and download the restrictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("area_config", type=Path)
    arguments = parser.parse_args()
    metadata = download_restrictions(load_study_area(arguments.area_config), RAW_DATA_DIRECTORY)
    print(f"Stored {metadata.relation_count} restrictions from {metadata.endpoint}")
    print(f"Overpass data timestamp: {metadata.osm_base_timestamp}")


if __name__ == "__main__":
    main()
