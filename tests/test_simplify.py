import networkx as nx

from graph_llm.graph.simplify import simplify_keeping_roads_apart
from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.restriction import TurnRestriction


def chain(names: list[str]) -> RoadGraph:
    """Two-way chain 0 - 1 - ... - n with one OSM way per edge; node 0 and n are dead ends."""
    graph: RoadGraph = nx.MultiDiGraph(crs="EPSG:4326", simplified=False)
    for node in range(len(names) + 1):
        graph.add_node(node, x=106.7 + node * 0.001, y=10.77, street_count=1)
    for index, name in enumerate(names):
        for source, target in ((index, index + 1), (index + 1, index)):
            graph.add_edge(
                source,
                target,
                osmid=100 + index,
                name=name,
                highway="residential",
                oneway=False,
                reversed=source > target,
                length=100.0,
            )
    return graph


def test_node_is_kept_where_the_name_changes() -> None:
    simplified = simplify_keeping_roads_apart(chain(["Cầu A", "Cầu A", "Đường B"]), [])

    assert set(simplified.nodes) == {0, 2, 3}
    names = {data["name"] for _, _, data in simplified.edges(data=True)}
    assert names == {"Cầu A", "Đường B"}


def test_node_of_a_restriction_is_kept() -> None:
    restriction = TurnRestriction(
        relation_id=1,
        kind="no_u_turn",
        condition=None,
        exempt_vehicles=[],
        vehicle_specific=None,
        from_way_ids=[100],
        via_node_ids=[1],
        via_way_ids=[],
        to_way_ids=[100],
    )

    simplified = simplify_keeping_roads_apart(chain(["Lê Lợi", "Lê Lợi"]), [restriction])

    assert 1 in simplified.nodes


def test_uniform_road_is_merged() -> None:
    simplified = simplify_keeping_roads_apart(chain(["Lê Lợi", "Lê Lợi", "Lê Lợi"]), [])

    assert set(simplified.nodes) == {0, 3}
