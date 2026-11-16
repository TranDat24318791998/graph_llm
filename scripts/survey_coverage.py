"""Survey the attribute coverage of a frozen graph snapshot.

Writes the report as JSON to data/processed/<area_id>/coverage.json and as
Markdown to docs/reports/osm_coverage_<area_id>.md.

Usage:
    uv run python scripts/survey_coverage.py hcmc_district1
"""

import argparse
from pathlib import Path

from graph_llm.graph.coverage import compute_coverage_report, render_coverage_markdown
from graph_llm.graph.snapshot import load_snapshot

RAW_DATA_DIRECTORY = Path("data/raw")
PROCESSED_DATA_DIRECTORY = Path("data/processed")
REPORT_DIRECTORY = Path("docs/reports")


def main() -> None:
    """Parse arguments, compute the coverage report and write it to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("area_id", help="Id of a downloaded study area.")
    arguments = parser.parse_args()

    graph, _ = load_snapshot(RAW_DATA_DIRECTORY, arguments.area_id)
    report = compute_coverage_report(graph, arguments.area_id)

    json_path = PROCESSED_DATA_DIRECTORY / arguments.area_id / "coverage.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    markdown_path = REPORT_DIRECTORY / f"osm_coverage_{arguments.area_id}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_coverage_markdown(report), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
