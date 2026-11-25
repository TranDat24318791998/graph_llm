"""Schemas of OSM turn restrictions, frozen next to a road graph snapshot."""

from datetime import datetime

from pydantic import Field

from graph_llm.schemas.base import StrictModel


class TurnRestriction(StrictModel):
    """One OSM restriction relation (type=restriction), as tagged in OSM.

    A prohibitory restriction (``no_*``) forbids the turn from ``from_way_ids`` to
    ``to_way_ids`` through the via node or via ways. A mandatory one (``only_*``)
    forbids every other turn out of the from way at the via point.
    """

    relation_id: int
    kind: str = Field(description="For example no_left_turn, no_u_turn, only_straight_on.")
    condition: str | None = Field(
        description="Time condition of a restriction:conditional tag; None if it always applies."
    )
    exempt_vehicles: list[str] = Field(description="Vehicle types listed in the except tag.")
    vehicle_specific: str | None = Field(
        description="Vehicle of a vehicle-specific tag such as restriction:motorcar, if any."
    )
    from_way_ids: list[int]
    via_node_ids: list[int]
    via_way_ids: list[int]
    to_way_ids: list[int]

    @property
    def applies_to_cars(self) -> bool:
        """Return True if the restriction applies to private cars."""
        if "motorcar" in self.exempt_vehicles:
            return False
        return self.vehicle_specific in (None, "motorcar", "motor_vehicle", "vehicle")

    @property
    def is_unconditional(self) -> bool:
        """Return True if the restriction applies at all times."""
        return self.condition is None


class RestrictionSnapshotMetadata(StrictModel):
    """Provenance of one frozen download of turn restrictions."""

    area_id: str
    downloaded_at: datetime
    endpoint: str
    query: str
    osm_base_timestamp: str = Field(description="Data timestamp reported by the Overpass server.")
    relation_count: int
    raw_response_sha256: str
    restrictions_sha256: str
