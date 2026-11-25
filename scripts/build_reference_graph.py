"""Build and freeze a reference graph (graph v2) and write its review report.

Writes data/processed/<graph_version>/ and docs/reports/graph_review_<graph_version>.md.
With --dry-run nothing is frozen: only the review report is written, so the review can
be iterated before the version is frozen.

Usage:
    uv run python scripts/build_reference_graph.py configs/graphs/hcmc_district1_v3_p2.yaml
    uv run python scripts/build_reference_graph.py <config> --dry-run
"""

import argparse
from pathlib import Path

from graph_llm.graph.carriageways import classify_all_streets
from graph_llm.graph.reference_graph import (
    build_reference_graph,
    load_reference_config,
    load_review_log,
    save_reference_graph,
)
from graph_llm.graph.review import ReviewInputs, build_review_cards
from graph_llm.graph.review_report import render_review_report

RAW_DATA_DIRECTORY = Path("data/raw")
PROCESSED_DATA_DIRECTORY = Path("data/processed")
REPORT_DIRECTORY = Path("docs/reports")


def main() -> None:
    """Parse arguments, build the graph, write the review report and freeze the graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    config = load_reference_config(arguments.config)
    graph, street_layer, junctions, restrictions, metadata = build_reference_graph(
        config, RAW_DATA_DIRECTORY
    )
    labels = classify_all_streets(
        graph, street_layer.streets, config.junctions.carriageway_max_separation_m
    )
    street_cards, junction_cards = build_review_cards(
        ReviewInputs(
            graph=graph,
            streets=street_layer.streets,
            labels=labels,
            junctions=junctions,
            restrictions=restrictions,
            area=metadata.source_snapshot.area.bounding_box,
        )
    )
    decisions = {}
    if config.review_log_path and Path(config.review_log_path).exists():
        log = load_review_log(Path(config.review_log_path))
        decisions = {decision.item_id: decision for decision in log.decisions}
    version = config.streets.graph_version
    report_path = REPORT_DIRECTORY / f"graph_review_{version}.md"
    report_path.write_text(
        render_review_report(
            version, street_cards, junction_cards, decisions, f"map_{version}.html"
        ),
        encoding="utf-8",
    )
    print(
        f"{version}: {metadata.node_count} nodes, {metadata.edge_count} edges, "
        f"{metadata.street_count} streets, {metadata.place_count} places, "
        f"{metadata.junction_count} junctions"
    )
    print(
        f"Review cards: {len(street_cards)} streets, {len(junction_cards)} junctions; "
        f"reviewed {sum(c.street_id in decisions for c in street_cards)} + "
        f"{sum(c.junction_id in decisions for c in junction_cards)}"
    )
    print(f"Wrote {report_path}")
    if arguments.dry_run:
        print("Dry run: the graph was not frozen.")
        return
    final = save_reference_graph(
        graph, street_layer.streets, junctions, metadata, PROCESSED_DATA_DIRECTORY
    )
    print(f"Froze {version} (graph sha256 {final.graphml_sha256[:12]}...)")


if __name__ == "__main__":
    main()
