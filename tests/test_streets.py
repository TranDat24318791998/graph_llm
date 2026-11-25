import networkx as nx
import pytest

from graph_llm.graph.streets import build_street_layer
from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.question import EdgeRef
from graph_llm.schemas.street import (
    GraphProcessingConfig,
    NameAliases,
    OverrideDecision,
    Street,
    StreetCategory,
    StreetOverride,
)


def add_two_way(
    graph: RoadGraph, source: int, target: int, length: float, name: str | None
) -> None:
    attributes: dict[str, object] = {"length": length}
    if name is not None:
        attributes["name"] = name
    graph.add_edge(source, target, **attributes)
    graph.add_edge(target, source, **attributes)


def build_hand_checked_graph() -> RoadGraph:
    """A line of eight nodes, south (1) to north (8).

    1-2  Le Loi, 100 m           \\ joined by a 150 m unnamed gap:
    2-3  unnamed, 150 m           > one street
    3-4  "Duong Le Loi", 100 m   /
    4-5  unnamed, 500 m
    5-6  Pasteur, 100 m          \\ 500 m unnamed gap: more than 200 m,
    6-7  unnamed, 500 m           > the automatic rule keeps them apart
    7-8  Pasteur, 100 m          /
    """
    graph: RoadGraph = nx.MultiDiGraph()
    for node in range(1, 9):
        graph.add_node(node, x=106.7, y=10.7 + node * 0.001)
    add_two_way(graph, 1, 2, 100.0, "Lê Lợi")
    add_two_way(graph, 2, 3, 150.0, None)
    add_two_way(graph, 3, 4, 100.0, "Đường Lê Lợi")
    add_two_way(graph, 4, 5, 500.0, None)
    add_two_way(graph, 5, 6, 100.0, "Pasteur")
    add_two_way(graph, 6, 7, 500.0, None)
    add_two_way(graph, 7, 8, 100.0, "Pasteur")
    return graph


def make_config(
    overrides: list[StreetOverride] | None = None,
    internal_road_names: list[str] | None = None,
    name_aliases: list[NameAliases] | None = None,
) -> GraphProcessingConfig:
    return GraphProcessingConfig(
        graph_version="test_p1",
        source_area_id="test_area",
        merge_gap_m=200.0,
        internal_road_names=internal_road_names or [],
        street_overrides=overrides or [],
        name_aliases=name_aliases or [],
    )


def pasteur_override(decision: OverrideDecision) -> StreetOverride:
    return StreetOverride(name="Pasteur", decision=decision, reason="test")


def street_by_id(streets: list[Street], street_id: str) -> Street:
    return next(street for street in streets if street.street_id == street_id)


def node_pairs(street: Street) -> set[tuple[int, int]]:
    return {(edge.source_node_id, edge.target_node_id) for edge in street.edges}


def test_name_without_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="pasteur"):
        build_street_layer(build_hand_checked_graph(), make_config())


def test_pieces_within_the_gap_become_one_street() -> None:
    layer = build_street_layer(
        build_hand_checked_graph(), make_config([pasteur_override(OverrideDecision.MERGE)])
    )
    le_loi = street_by_id(layer.streets, "le-loi")

    assert le_loi.display_name == "Lê Lợi"
    assert le_loi.osm_names == ["Lê Lợi", "Đường Lê Lợi"]
    assert node_pairs(le_loi) == {(1, 2), (2, 1), (3, 4), (4, 3)}
    assert not le_loi.shares_name_with_other_street
    assert layer.unnamed_edge_count == 6


def test_merge_override_makes_one_street() -> None:
    layer = build_street_layer(
        build_hand_checked_graph(), make_config([pasteur_override(OverrideDecision.MERGE)])
    )
    pasteur = street_by_id(layer.streets, "pasteur")

    assert node_pairs(pasteur) == {(5, 6), (6, 5), (7, 8), (8, 7)}
    assert [street.street_id for street in layer.streets] == ["le-loi", "pasteur"]


def test_separate_override_makes_numbered_streets_from_north_to_south() -> None:
    layer = build_street_layer(
        build_hand_checked_graph(), make_config([pasteur_override(OverrideDecision.SEPARATE)])
    )
    northern = street_by_id(layer.streets, "pasteur-1")
    southern = street_by_id(layer.streets, "pasteur-2")

    assert node_pairs(northern) == {(7, 8), (8, 7)}
    assert node_pairs(southern) == {(5, 6), (6, 5)}
    assert northern.shares_name_with_other_street
    assert southern.shares_name_with_other_street


def test_resolutions_record_automatic_and_manual_decisions() -> None:
    layer = build_street_layer(
        build_hand_checked_graph(), make_config([pasteur_override(OverrideDecision.SEPARATE)])
    )
    resolutions = {resolution.name_key: resolution for resolution in layer.name_resolutions}

    assert resolutions["lê lợi"].override is None
    assert resolutions["lê lợi"].groups_after_automatic_rule == 1
    assert resolutions["pasteur"].override is OverrideDecision.SEPARATE
    assert resolutions["pasteur"].street_ids == ["pasteur-1", "pasteur-2"]


def test_unnecessary_override_is_rejected() -> None:
    overrides = [
        pasteur_override(OverrideDecision.MERGE),
        StreetOverride(name="Lê Lợi", decision=OverrideDecision.MERGE, reason="test"),
    ]
    with pytest.raises(ValueError, match="unnecessary"):
        build_street_layer(build_hand_checked_graph(), make_config(overrides))


def test_unknown_internal_road_name_is_rejected() -> None:
    config = make_config(
        [pasteur_override(OverrideDecision.MERGE)], internal_road_names=["Riverview 3"]
    )
    with pytest.raises(ValueError, match="riverview 3"):
        build_street_layer(build_hand_checked_graph(), config)


def test_alias_that_is_another_street_name_is_rejected() -> None:
    config = make_config(
        [pasteur_override(OverrideDecision.MERGE)],
        name_aliases=[NameAliases(name="Lê Lợi", aliases=["Pasteur"])],
    )
    with pytest.raises(ValueError, match="another street"):
        build_street_layer(build_hand_checked_graph(), config)


def test_aliases_are_attached_to_their_street() -> None:
    config = make_config(
        [pasteur_override(OverrideDecision.MERGE)],
        name_aliases=[NameAliases(name="Đường Lê Lợi", aliases=["LL"])],
    )
    layer = build_street_layer(build_hand_checked_graph(), config)

    assert street_by_id(layer.streets, "le-loi").aliases == ["LL"]
    assert street_by_id(layer.streets, "pasteur").aliases == []


def test_edge_with_several_names_belongs_to_each_street_and_categories_are_set() -> None:
    graph: RoadGraph = nx.MultiDiGraph()
    graph.add_node(1, x=106.7, y=10.7)
    graph.add_node(2, x=106.7, y=10.701)
    graph.add_edge(1, 2, length=100.0, name=["Aqua 1", "Hẻm 5 Pasteur"])
    layer = build_street_layer(graph, make_config(internal_road_names=["Aqua 1"]))

    categories = {street.street_id: street.category for street in layer.streets}
    assert categories == {"aqua-1": StreetCategory.INTERNAL, "hem-5-pasteur": StreetCategory.ALLEY}
    for street in layer.streets:
        assert street.edges == [EdgeRef(source_node_id=1, target_node_id=2, key=0)]
