"""Write an interactive web map of a reference graph (graph v2) with its review flags.

The graph is built in memory from its config, so the map can be drawn before the
version is frozen (during the review). Writes docs/reports/map_<graph_version>.html.
The page uses folium (Leaflet); base map tiles are loaded from the web when it is opened.

Usage:
    uv run python scripts/render_map.py configs/graphs/hcmc_district1_v3_p2.yaml
"""

import argparse
from pathlib import Path

from graph_llm.graph.carriageways import classify_all_streets
from graph_llm.graph.map_view import FlaggedItem, build_map
from graph_llm.graph.reference_graph import (
    build_reference_graph,
    excluded_item_ids,
    load_reference_config,
    load_review_log,
)
from graph_llm.graph.review import ReviewInputs, build_review_cards
from graph_llm.questions.graph_view import GraphView

RAW_DATA_DIRECTORY = Path("data/raw")
REPORT_DIRECTORY = Path("docs/reports")


def main() -> None:
    """Parse arguments, build the graph in memory and write the map page."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()

    config = load_reference_config(arguments.config)
    graph, street_layer, junctions, restrictions, metadata = build_reference_graph(
        config, RAW_DATA_DIRECTORY
    )
    review_log = None
    if config.review_log_path and Path(config.review_log_path).exists():
        review_log = load_review_log(Path(config.review_log_path))
    view = GraphView.build(
        graph,
        street_layer.streets,
        config.junctions,
        restrictions,
        junctions,
        excluded_item_ids(review_log),
    )
    area = metadata.source_snapshot.area.bounding_box
    street_cards, junction_cards = build_review_cards(
        ReviewInputs(
            graph=graph,
            streets=street_layer.streets,
            labels=classify_all_streets(
                graph, street_layer.streets, config.junctions.carriageway_max_separation_m
            ),
            junctions=junctions,
            restrictions=restrictions,
            area=area,
        )
    )
    flagged = [
        FlaggedItem(card.name, card.latitude, card.longitude, ", ".join(f.code for f in card.flags))
        for card in street_cards
        if card.flags
    ] + [
        FlaggedItem(
            card.junction_id,
            card.latitude,
            card.longitude,
            ", ".join(f.code for f in card.flags),
        )
        for card in junction_cards
        if card.flags
    ]
    version = config.streets.graph_version
    world = build_map(view, version, area, flagged)
    output_path = REPORT_DIRECTORY / f"map_{version}.html"
    world.save(str(output_path))
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
