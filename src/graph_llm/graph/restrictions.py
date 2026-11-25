"""Download OSM turn restrictions once, freeze them, and map them onto graph edges.

OSMnx does not read restriction relations, so they are fetched separately with an
Overpass query over the same rectangle as the snapshot. Like a snapshot, the
download is stored with its provenance and never repeated for the same id. The
Overpass data date is recorded, because it can differ from the snapshot date.

A restriction names OSM ways; graph edges carry the ids of the ways they were
built from (``osmid``, a list after simplification). A turn is the pair (edge
into the via node, edge out of it), so restrictions become sets of forbidden
edge pairs at a node. Only restrictions with a single via node are mapped; ones
whose via is a way are reported and left out of routing.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from graph_llm.graph.snapshot import compute_sha256
from graph_llm.graph.types import RoadGraph
from graph_llm.schemas.graph import StudyArea
from graph_llm.schemas.question import EdgeRef
from graph_llm.schemas.restriction import RestrictionSnapshotMetadata, TurnRestriction

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = "graph-llm-research-client/0.1"
RAW_FILE_NAME = "overpass_response.json"
RESTRICTIONS_FILE_NAME = "restrictions.json"
METADATA_FILE_NAME = "metadata.json"
CONDITIONAL_KEY = "restriction:conditional"


def restriction_directory(raw_data_directory: Path, area_id: str) -> Path:
    """Return the directory of the frozen restrictions of one snapshot area."""
    return raw_data_directory / f"{area_id}_restrictions"


def overpass_query(area: StudyArea) -> str:
    """Return the Overpass query for all restriction relations in the area's rectangle."""
    box = area.bounding_box
    return (
        "[out:json][timeout:120];"
        f'relation["type"="restriction"]({box.south},{box.west},{box.north},{box.east});'
        "out body;"
    )


def split_values(value: str | None) -> list[str]:
    """Split an OSM multi-value tag (separated by ';' or ',') into clean items."""
    if not value:
        return []
    return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]


def parse_restriction(element: dict[str, object]) -> TurnRestriction | None:
    """Turn one Overpass relation element into a restriction; None if it has no restriction."""
    tags = element.get("tags")
    members = element.get("members")
    if not isinstance(tags, dict) or not isinstance(members, list):
        return None
    kind: str | None = tags.get("restriction")
    condition: str | None = None
    vehicle: str | None = None
    if kind is None and CONDITIONAL_KEY in tags:
        value = str(tags[CONDITIONAL_KEY])
        kind, _, rest = value.partition("@")
        kind = kind.strip()
        condition = rest.strip().strip("()").strip() or None
    if kind is None:
        for key, value in tags.items():
            if key.startswith("restriction:") and not key.endswith(":conditional"):
                vehicle = key.split(":", 1)[1]
                kind = str(value)
                break
    if kind is None:
        return None
    ways: dict[str, list[int]] = {"from": [], "to": [], "via_way": [], "via_node": []}
    for member in members:
        if not isinstance(member, dict):
            continue
        role = member.get("role")
        member_type = member.get("type")
        reference = int(str(member.get("ref")))
        if role in ("from", "to") and member_type == "way":
            ways[str(role)].append(reference)
        elif role == "via" and member_type == "node":
            ways["via_node"].append(reference)
        elif role == "via" and member_type == "way":
            ways["via_way"].append(reference)
    return TurnRestriction(
        relation_id=int(str(element["id"])),
        kind=kind,
        condition=condition,
        exempt_vehicles=split_values(tags.get("except")),
        vehicle_specific=vehicle,
        from_way_ids=ways["from"],
        via_node_ids=ways["via_node"],
        via_way_ids=ways["via_way"],
        to_way_ids=ways["to"],
    )


def fetch_overpass(query: str, rounds: int = 4, first_wait_s: float = 30.0) -> tuple[str, bytes]:
    """Run a query on the first Overpass endpoint that answers; return endpoint and body.

    Public Overpass servers are often busy; every endpoint is tried in turn, and the
    whole round is repeated with exponential backoff, as the servers ask.
    """
    errors = []
    wait_s = first_wait_s
    for round_index in range(rounds):
        for endpoint in OVERPASS_ENDPOINTS:
            request = urllib.request.Request(
                endpoint,
                data=urllib.parse.urlencode({"data": query}).encode(),
                headers={"User-Agent": USER_AGENT},
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    body: bytes = response.read()
                json.loads(body)
                return endpoint, body
            except (OSError, ValueError) as error:
                errors.append(f"round {round_index + 1}, {endpoint}: {error}")
        if round_index < rounds - 1:
            time.sleep(wait_s)
            wait_s *= 2
    raise RuntimeError("No Overpass endpoint answered: " + "; ".join(errors))


def download_restrictions(area: StudyArea, raw_data_directory: Path) -> RestrictionSnapshotMetadata:
    """Download and freeze the restriction relations of an area.

    Raises:
        FileExistsError: If restrictions for this area id are already stored.
    """
    target = restriction_directory(raw_data_directory, area.area_id)
    if target.exists():
        raise FileExistsError(f"Restrictions already exist at {target}; they are immutable.")
    query = overpass_query(area)
    endpoint, body = fetch_overpass(query)
    response = json.loads(body)
    elements = response.get("elements", [])
    restrictions = [r for r in (parse_restriction(e) for e in elements) if r is not None]
    target.mkdir(parents=True)
    (target / RAW_FILE_NAME).write_bytes(body)
    restrictions_path = target / RESTRICTIONS_FILE_NAME
    restrictions_path.write_text(
        json.dumps([r.model_dump() for r in restrictions], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    metadata = RestrictionSnapshotMetadata(
        area_id=area.area_id,
        downloaded_at=datetime.now(UTC),
        endpoint=endpoint,
        query=query,
        osm_base_timestamp=str(response.get("osm3s", {}).get("timestamp_osm_base", "")),
        relation_count=len(restrictions),
        raw_response_sha256=compute_sha256(target / RAW_FILE_NAME),
        restrictions_sha256=compute_sha256(restrictions_path),
    )
    (target / METADATA_FILE_NAME).write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return metadata


def load_restrictions(
    raw_data_directory: Path, area_id: str
) -> tuple[list[TurnRestriction], RestrictionSnapshotMetadata]:
    """Load frozen restrictions and verify their checksum."""
    target = restriction_directory(raw_data_directory, area_id)
    metadata = RestrictionSnapshotMetadata.model_validate_json(
        (target / METADATA_FILE_NAME).read_text(encoding="utf-8")
    )
    restrictions_path = target / RESTRICTIONS_FILE_NAME
    if compute_sha256(restrictions_path) != metadata.restrictions_sha256:
        raise ValueError(f"{restrictions_path} does not match its recorded checksum.")
    items = json.loads(restrictions_path.read_text(encoding="utf-8"))
    return [TurnRestriction.model_validate(item) for item in items], metadata


def way_ids_of_edge(graph: RoadGraph, edge: EdgeRef) -> set[int]:
    """Return the OSM way ids an edge was built from."""
    raw = graph.edges[edge.source_node_id, edge.target_node_id, edge.key].get("osmid")
    values: Iterable[object] = raw if isinstance(raw, list) else [raw]
    return {int(str(value)) for value in values if value is not None}


class MappingStatus(StrEnum):
    """What became of one restriction when it was mapped onto graph edges."""

    MAPPED = "mapped"
    OUTSIDE_GRAPH = "outside_graph"
    ALREADY_IMPOSSIBLE = "already_impossible"
    UNMAPPABLE = "unmappable"


@dataclass(frozen=True)
class MappedRestriction:
    """A restriction expressed as forbidden edge sequences.

    A via-node restriction gives sequences of two edges (in-edge, out-edge). A
    via-way restriction gives longer sequences: the in-edge, the edges along the
    via way(s), and the out-edge. A route is illegal if it contains one of the
    sequences as consecutive edges.
    """

    restriction: TurnRestriction
    status: MappingStatus
    forbidden_sequences: frozenset[tuple[EdgeRef, ...]]


def edges_of_ways(graph: RoadGraph, way_ids: set[int]) -> list[EdgeRef]:
    """Return the graph edges built from any of the given OSM ways."""
    return [
        EdgeRef(source_node_id=source, target_node_id=target, key=key)
        for source, target, key in graph.edges(keys=True)
        if way_ids_of_edge(graph, EdgeRef(source_node_id=source, target_node_id=target, key=key))
        & way_ids
    ]


def via_paths(
    graph: RoadGraph, start: int, via_edges: list[EdgeRef], limit: int = 6
) -> list[tuple[list[EdgeRef], int]]:
    """Return the directed walks along via edges from a start node (with their end node)."""
    outgoing: dict[int, list[EdgeRef]] = {}
    for edge in via_edges:
        outgoing.setdefault(edge.source_node_id, []).append(edge)
    walks: list[tuple[list[EdgeRef], int]] = [([], start)]
    result: list[tuple[list[EdgeRef], int]] = []
    while walks:
        path, node = walks.pop()
        if path:
            result.append((path, node))
        if len(path) >= limit:
            continue
        for edge in outgoing.get(node, []):
            if edge not in path:
                walks.append((path + [edge], edge.target_node_id))
    return result


def map_restriction(graph: RoadGraph, restriction: TurnRestriction) -> MappedRestriction:
    """Map a restriction onto graph edges as forbidden edge sequences.

    Status:
    - ``OUTSIDE_GRAPH``: the via node, the from way or the to way is not in the graph
      (outside the area, pruned, or not drivable), so it cannot affect routing here.
    - ``ALREADY_IMPOSSIBLE``: everything is in the graph, but the turn cannot be made
      anyway (for example the to way is one-way towards the via point).
    - ``UNMAPPABLE``: malformed relation (no via, or several via nodes).
    - ``MAPPED``: at least one forbidden sequence.
    """
    from_ways = set(restriction.from_way_ids)
    to_ways = set(restriction.to_way_ids)
    from_edges = edges_of_ways(graph, from_ways)
    to_edges = edges_of_ways(graph, to_ways)
    if restriction.via_way_ids:
        if restriction.kind.startswith("only_"):
            # Mandatory turns through a via way do not occur in the study area.
            return MappedRestriction(restriction, MappingStatus.UNMAPPABLE, frozenset())
        via_edges = edges_of_ways(graph, set(restriction.via_way_ids))
        if not from_edges or not to_edges or not via_edges:
            return MappedRestriction(restriction, MappingStatus.OUTSIDE_GRAPH, frozenset())
        via_nodes = {e.source_node_id for e in via_edges} | {e.target_node_id for e in via_edges}
        sequences: set[tuple[EdgeRef, ...]] = set()
        for entering in from_edges:
            if entering.target_node_id not in via_nodes:
                continue
            for path, end in via_paths(graph, entering.target_node_id, via_edges):
                for leaving in to_edges:
                    if leaving.source_node_id == end and leaving not in path:
                        sequences.add((entering, *path, leaving))
        status = MappingStatus.MAPPED if sequences else MappingStatus.ALREADY_IMPOSSIBLE
        return MappedRestriction(restriction, status, frozenset(sequences))
    if len(restriction.via_node_ids) != 1:
        return MappedRestriction(restriction, MappingStatus.UNMAPPABLE, frozenset())
    node = restriction.via_node_ids[0]
    if node not in graph or not from_edges or not to_edges:
        return MappedRestriction(restriction, MappingStatus.OUTSIDE_GRAPH, frozenset())
    into_via = [e for e in from_edges if e.target_node_id == node]
    out_to = [e for e in to_edges if e.source_node_id == node]
    if not into_via or not out_to:
        return MappedRestriction(restriction, MappingStatus.ALREADY_IMPOSSIBLE, frozenset())
    forbidden: set[tuple[EdgeRef, ...]]
    if restriction.kind.startswith("only_"):
        out_edges = [
            EdgeRef(source_node_id=node, target_node_id=target, key=key)
            for _, target, key in graph.out_edges(node, keys=True)
        ]
        forbidden = {(e, o) for e in into_via for o in out_edges if o not in set(out_to)}
    else:
        forbidden = {(e, o) for e in into_via for o in out_to}
    return MappedRestriction(restriction, MappingStatus.MAPPED, frozenset(forbidden))
