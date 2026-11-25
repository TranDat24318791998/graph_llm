"""Render the processed graph as an interactive web map (folium / Leaflet).

The road data drawn on the map is the frozen processed graph used for the
evaluation. The base map (Esri tiles built on OSM and other sources, loaded from the web)
only gives orientation; it shows today's map data and may differ slightly from the
frozen snapshot.

One drawn line stands for one pair of nodes: a two-way pair is drawn once, a
one-way edge is drawn in its own colour and gets a small arrow at its midpoint in
the legal direction. Arrows are geometry in metres, so they stay small when zoomed
out and become readable when zoomed in.

Lines are drawn with an estimated real width in metres (lane count times
``LANE_WIDTH_M`` when OSM has it, otherwise a typical width for the road class),
converted to pixels at every zoom level, with a minimum width so that the network
stays visible when zoomed out. Street names are written along the lines from zoom
level ``LABEL_MINIMUM_ZOOM`` on, only where the segment is long enough on screen
for the text, as on common web maps. Widths are estimates for display only; they
are not part of the evaluation data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import folium
from branca.element import Figure, MacroElement
from folium.elements import JSCSSMixin
from folium.plugins import Fullscreen, Search
from jinja2 import Template

from graph_llm.questions.graph_view import GraphView
from graph_llm.schemas.graph import BoundingBox
from graph_llm.schemas.question import EdgeRef
from graph_llm.schemas.street import StreetCategory

COORDINATE_DIGITS = 6
METRES_PER_DEGREE_LATITUDE = 111_320.0
ARROW_MINIMUM_SEGMENT_M = 40.0
ARROW_ARM_M = 7.0
ARROW_HALF_ANGLE_DEGREES = 28.0
LANE_WIDTH_M = 3.2
MAXIMUM_ROAD_WIDTH_M = 24.0
ALLEY_WIDTH_M = 3.5
INTERNAL_ROAD_WIDTH_M = 5.0
# Typical carriageway widths by OSM highway class, used when the lane count is unknown.
ROAD_WIDTH_BY_HIGHWAY_M = {
    "trunk": 16.0,
    "primary": 14.0,
    "secondary": 12.0,
    "tertiary": 10.0,
    "unclassified": 7.0,
    "residential": 7.0,
    "living_street": 5.0,
    "service": 4.0,
    "trunk_link": 6.0,
    "primary_link": 6.0,
    "secondary_link": 6.0,
    "tertiary_link": 5.0,
}
DEFAULT_ROAD_WIDTH_M = 7.0
LABEL_MINIMUM_ZOOM = 16
TEXTPATH_SCRIPT_URL = "https://cdn.jsdelivr.net/npm/leaflet-textpath@1.2.3/leaflet.textpath.min.js"

COLOURS = {
    "two_way": "#3b6ea5",
    "one_way": "#e8590c",
    "alley": "#8d8d86",
    "internal": "#2f9e44",
    "unnamed": "#868e96",
    "arrow": "#7c2d12",
    "unnamed_arrow": "#343a40",
    "askable": "#c2255c",
    "area": "#495057",
}
ONEWAY_STATUS_TEXT = {
    "one_way": "one-way along its whole length",
    "two_way": "two-way along its whole length",
    "two_way_divided": "two-way, with a median on some or all parts",
    "mixed": "mixed one-way and two-way parts",
}
JUNCTION_COLOURS = {
    "single_node": "#495057",
    "compact": "#1971c2",
    "divided_crossing": "#7048e8",
    "with_connectors": "#e67700",
    "roundabout_or_place": "#2b8a3e",
}
UNNAMED_LABEL = "(unnamed edge)"
TOOLTIP_FIELDS = ["name", "category", "street", "segment", "askable"]
TOOLTIP_ALIASES = ["Street", "Category", "Whole street", "These lines", "Askable in Phase 1"]


@dataclass(frozen=True)
class MapSegment:
    """One drawn line: an edge, or both edges of a two-way pair."""

    points: list[tuple[float, float]]
    street_ids: list[str]
    one_way: bool
    length_m: int
    width_m: float


def edge_points(view: GraphView, edge: EdgeRef) -> list[tuple[float, float]]:
    """Return the (longitude, latitude) points of an edge, from its geometry if it has one."""
    geometry = view.edge_attributes(edge).get("geometry")
    if geometry is not None and hasattr(geometry, "coords"):
        points = [(float(x), float(y)) for x, y in geometry.coords]
    else:
        source = view.graph.nodes[edge.source_node_id]
        target = view.graph.nodes[edge.target_node_id]
        points = [
            (float(source["x"]), float(source["y"])),
            (float(target["x"]), float(target["y"])),
        ]
    return [(round(x, COORDINATE_DIGITS), round(y, COORDINATE_DIGITS)) for x, y in points]


def first_value(value: object) -> str | None:
    """Return a single attribute value; for a list (merged OSM ways) the first item."""
    if isinstance(value, list):
        return str(value[0]) if value else None
    return None if value is None else str(value)


def road_width_m(view: GraphView, edge: EdgeRef) -> float:
    """Estimate the carriageway width of an edge in metres, for display only."""
    owners = view.street_ids_of_edge.get(edge, [])
    categories = {view.streets[o].category for o in owners}
    if StreetCategory.ALLEY in categories:
        return ALLEY_WIDTH_M
    if StreetCategory.INTERNAL in categories:
        return INTERNAL_ROAD_WIDTH_M
    attributes = view.edge_attributes(edge)
    raw_lanes = attributes.get("lanes")
    lane_values = raw_lanes if isinstance(raw_lanes, list) else [raw_lanes]
    lanes = [
        float(str(v)) for v in lane_values if v is not None and str(v).replace(".", "").isdigit()
    ]
    if lanes:
        return min(MAXIMUM_ROAD_WIDTH_M, max(lanes) * LANE_WIDTH_M)
    highway = first_value(attributes.get("highway"))
    return ROAD_WIDTH_BY_HIGHWAY_M.get(highway or "", DEFAULT_ROAD_WIDTH_M)


def map_segments(view: GraphView) -> list[MapSegment]:
    """Return one segment per edge, merging the two edges of a two-way pair of one street."""
    edges = sorted(
        (
            EdgeRef(source_node_id=u, target_node_id=v, key=k)
            for u, v, k in view.graph.edges(keys=True)
        ),
        key=lambda e: (e.source_node_id, e.target_node_id, e.key),
    )
    edge_set = set(edges)
    drawn: set[EdgeRef] = set()
    segments = []
    for edge in edges:
        if edge in drawn:
            continue
        drawn.add(edge)
        reverse = EdgeRef(
            source_node_id=edge.target_node_id, target_node_id=edge.source_node_id, key=edge.key
        )
        owners = view.street_ids_of_edge.get(edge, [])
        two_way = reverse in edge_set and view.street_ids_of_edge.get(reverse, []) == owners
        if two_way:
            drawn.add(reverse)
        segments.append(
            MapSegment(
                points=edge_points(view, edge),
                street_ids=list(owners),
                one_way=not two_way,
                length_m=round(view.edge_length_m(edge)),
                width_m=road_width_m(view, edge),
            )
        )
    return segments


def layer_of(view: GraphView, segment: MapSegment) -> str:
    """Return the map layer a segment belongs to: street, alley, internal or unnamed."""
    if not segment.street_ids:
        return "unnamed"
    category = view.streets[segment.street_ids[0]].category
    if category is StreetCategory.ALLEY:
        return "alley"
    if category is StreetCategory.INTERNAL:
        return "internal"
    return "street"


def segment_properties(view: GraphView, segment: MapSegment) -> dict[str, object]:
    """Return the tooltip fields, drawing direction and display width of a segment."""
    streets = [view.streets[s] for s in segment.street_ids]
    direction = "one_way" if segment.one_way else "two_way"
    description = f"{'one-way' if segment.one_way else 'two-way'}, {segment.length_m} m"
    if not streets:
        return {
            "name": UNNAMED_LABEL,
            "category": "-",
            "street": "-",
            "segment": description,
            "askable": "no",
            "direction": direction,
            "width_m": segment.width_m,
        }
    return {
        "name": " / ".join(s.display_name for s in streets),
        "category": ", ".join(s.category.value for s in streets),
        "street": "; ".join(ONEWAY_STATUS_TEXT[view.direction(s.street_id).value] for s in streets),
        "segment": description,
        "askable": "yes" if any(view.is_askable(s.street_id) for s in streets) else "no",
        "direction": direction,
        "width_m": segment.width_m,
    }


def line_feature(
    points: list[tuple[float, float]], properties: dict[str, object]
) -> dict[str, object]:
    """Return a GeoJSON LineString feature."""
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [list(p) for p in points]},
        "properties": properties,
    }


def point_along(points: list[tuple[float, float]], fraction: float) -> tuple[int, float, float]:
    """Return the piece index and the position at a fraction of a polyline's length."""
    scale_x = math.cos(math.radians(points[0][1]))
    pieces = [
        math.hypot((b[0] - a[0]) * scale_x, b[1] - a[1])
        for a, b in zip(points, points[1:], strict=False)
    ]
    if not pieces:
        raise ValueError("A polyline needs at least two points.")
    target = sum(pieces) * fraction
    for index, piece in enumerate(pieces):
        if target <= piece or index == len(pieces) - 1:
            t = 0.0 if piece == 0 else min(1.0, target / piece)
            a, b = points[index], points[index + 1]
            return index, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        target -= piece
    raise AssertionError("unreachable")


def arrow_points(
    points: list[tuple[float, float]], road_width_m: float
) -> list[tuple[float, float]]:
    """Return a chevron at the middle of a polyline, pointing along it, sized to the road."""
    index, x, y = point_along(points, 0.5)
    a, b = points[index], points[index + 1]
    scale_x = math.cos(math.radians(y))
    heading = math.atan2(b[1] - a[1], (b[0] - a[0]) * scale_x)
    arm = max(ARROW_ARM_M, 0.45 * road_width_m) / METRES_PER_DEGREE_LATITUDE
    chevron = []
    for side in (1, -1):
        angle = heading + math.pi - side * math.radians(ARROW_HALF_ANGLE_DEGREES)
        chevron.append((x + arm * math.cos(angle) / scale_x, y + arm * math.sin(angle)))
    return [chevron[0], (x, y), chevron[1]]


def reading_direction(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the points ordered so that text along them reads from left to right."""
    return points if points[-1][0] >= points[0][0] else list(reversed(points))


def collection(features: list[dict[str, object]]) -> dict[str, object]:
    """Return a GeoJSON FeatureCollection."""
    return {"type": "FeatureCollection", "features": features}


def tooltip() -> folium.GeoJsonTooltip:
    """Return the tooltip shown when hovering a road segment."""
    return folium.GeoJsonTooltip(fields=TOOLTIP_FIELDS, aliases=TOOLTIP_ALIASES, sticky=True)


def plain_layer(
    features: list[dict[str, object]],
    name: str,
    colour: str,
    show: bool,
    dash: str | None = None,
) -> folium.GeoJson:
    """Return a single-colour line layer with a tooltip; widths are set by ``ZoomStyling``."""
    style: dict[str, object] = {"color": colour, "opacity": 0.9, "lineCap": "butt"}
    if dash is not None:
        style["dashArray"] = dash
    return folium.GeoJson(
        collection(features),
        name=name,
        show=show,
        style_function=lambda _feature: style,
        highlight_function=lambda _feature: {"color": COLOURS["askable"], "opacity": 1},
        tooltip=tooltip(),
    )


def street_style(feature: dict[str, dict[str, str]]) -> dict[str, object]:
    """Colour a street segment by its direction; its width is set by ``ZoomStyling``."""
    return {
        "color": COLOURS[feature["properties"]["direction"]],
        "opacity": 0.85,
        "lineCap": "butt",
    }


def arrow_style(feature: dict[str, dict[str, str]]) -> dict[str, object]:
    """Draw arrows dark, so they stay visible on top of wide orange or grey lines."""
    unnamed = feature["properties"]["name"] == UNNAMED_LABEL
    return {"color": COLOURS["unnamed_arrow" if unnamed else "arrow"], "weight": 2, "opacity": 1}


class MetricScale(MacroElement):
    """A Leaflet scale bar in metres only."""

    def __init__(self) -> None:
        """Create the scale bar element."""
        super().__init__()  # type: ignore[no-untyped-call]
        self._name = "MetricScale"

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        L.control.scale({imperial: false, position: "bottomright"})
            .addTo({{ this._parent.get_name() }});
        {% endmacro %}
        """
    )


class ZoomStyling(JSCSSMixin):
    """Set line widths from metres at every zoom, and write street names along the lines.

    Performance: road lines are drawn on a canvas (``prefer_canvas``); only the street
    names use SVG, because Leaflet.TextPath needs it. Label lines are created lazily,
    only for segments on screen from ``LABEL_MINIMUM_ZOOM`` on, and only labels that
    change are added or removed. Updates after panning are coalesced into one frame.

    ``width_layers`` are (layer, minimum width in pixels) pairs whose features carry a
    ``width_m`` property. ``label_lines`` are (text, [[lat, lon], ...]) pairs, oriented
    left to right. The labels are added to ``layer_control`` as an overlay.
    """

    def __init__(
        self,
        width_layers: list[tuple[folium.GeoJson, float]],
        label_lines: list[tuple[str, list[list[float]]]],
        layer_control: folium.LayerControl,
        bounds: list[list[float]],
    ) -> None:
        """Create the element; add it to the map after all the layers it refers to."""
        super().__init__()  # type: ignore[no-untyped-call]
        self._name = "ZoomStyling"
        self.width_layers = width_layers
        self.label_json = json.dumps(label_lines, ensure_ascii=False, separators=(",", ":"))
        self.layer_control = layer_control
        self.bounds = json.dumps(bounds)
        self.label_minimum_zoom = LABEL_MINIMUM_ZOOM

    default_js = [("leaflet.textpath", TEXTPATH_SCRIPT_URL)]

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
          var map = {{ this._parent.get_name() }};
          var widthLayers = [
            {% for layer, minimum in this.width_layers %}
            [{{ layer.get_name() }}, {{ minimum }}],
            {% endfor %}
          ];
          var labelData = {{ this.label_json }};
          // Roads are on the canvas renderer that prefer_canvas created. Leaflet.TextPath
          // always writes into map._renderer, so that slot points to an SVG renderer used
          // only for labels, while map.options.renderer keeps every road layer (including
          // layers switched on later) on the canvas.
          var labelRenderer = L.svg({padding: 0.2});
          map.options.renderer = map._renderer || L.canvas();
          map._renderer = labelRenderer;
          var labelGroup = L.layerGroup().addTo(map);
          {{ this.layer_control.get_name() }}.addOverlay(
            labelGroup, "Street names (from zoom {{ this.label_minimum_zoom }})");
          var entries = labelData.map(function (item) {
            return {text: item[0], points: item[1], bounds: L.latLngBounds(item[1]), line: null};
          });
          var textStyle = {
            "font-family": "system-ui, -apple-system, sans-serif",
            "font-weight": "600", "fill": "#1f2933", "stroke": "#ffffff",
            "stroke-width": "3", "paint-order": "stroke", "pointer-events": "none"
          };

          function metresPerPixel() {
            var latitude = map.getCenter().lat * Math.PI / 180;
            return 156543.03392 * Math.cos(latitude) / Math.pow(2, map.getZoom());
          }
          function updateWidths() {
            var mpp = metresPerPixel();
            widthLayers.forEach(function (pair) {
              if (!map.hasLayer(pair[0])) { return; }
              pair[0].eachLayer(function (line) {
                var weight = Math.max(pair[1], line.feature.properties.width_m / mpp);
                if (line.options.weight !== weight) { line.setStyle({weight: weight}); }
              });
            });
          }
          function pixelLength(points) {
            var total = 0;
            var previous = map.latLngToLayerPoint(points[0]);
            for (var i = 1; i < points.length; i++) {
              var current = map.latLngToLayerPoint(points[i]);
              total += current.distanceTo(previous);
              previous = current;
            }
            return total;
          }
          function hideLabel(entry) {
            if (entry.line) { labelGroup.removeLayer(entry.line); entry.line = null; }
          }
          function updateLabels() {
            var zoom = map.getZoom();
            var active = map.hasLayer(labelGroup) && zoom >= {{ this.label_minimum_zoom }};
            var size = zoom >= 18 ? 14 : (zoom >= 17 ? 12.5 : 11);
            var visible = map.getBounds().pad(0.1);
            var placed = {};
            entries.forEach(function (entry) {
              var show = active && visible.intersects(entry.bounds)
                && pixelLength(entry.points) > entry.text.length * size * 0.6 + 24;
              if (show) {
                var centre = map.latLngToLayerPoint(entry.bounds.getCenter());
                var others = placed[entry.text] || [];
                for (var k = 0; k < others.length; k++) {
                  if (others[k].distanceTo(centre) < 260) { show = false; break; }
                }
                if (show) { others.push(centre); placed[entry.text] = others; }
              }
              if (!show) { hideLabel(entry); return; }
              if (entry.line && entry.size === size) { return; }
              hideLabel(entry);
              entry.line = L.polyline(entry.points, {
                renderer: labelRenderer, opacity: 0, weight: 1, interactive: false
              });
              labelGroup.addLayer(entry.line);
              entry.size = size;
              entry.line.setText(entry.text, {
                center: true, offset: size * 0.35,
                attributes: Object.assign({"font-size": size + "px"}, textStyle)
              });
            });
          }
          var pending = false;
          function scheduleLabels() {
            if (pending) { return; }
            pending = true;
            window.requestAnimationFrame(function () { pending = false; updateLabels(); });
          }
          map.on("zoomend", function () { updateWidths(); scheduleLabels(); });
          map.on("moveend", scheduleLabels);
          map.on("overlayadd", function () { updateWidths(); scheduleLabels(); });
          map.on("overlayremove", scheduleLabels);
          updateWidths();
          updateLabels();

          // Open the map at "#latitude,longitude,zoom" (links in the review report) and
          // circle that point.
          var focusMarker = null;
          function viewFromHash() {
            var match = window.location.hash.match(/^#(-?[0-9.]+),(-?[0-9.]+)(?:,([0-9.]+))?/);
            if (!match) { return; }
            var point = [parseFloat(match[1]), parseFloat(match[2])];
            map.setView(point, match[3] ? parseFloat(match[3]) : 18, {animate: false});
            if (focusMarker) { map.removeLayer(focusMarker); }
            focusMarker = L.circleMarker(point, {
              radius: 22, color: "#c2255c", weight: 3, fill: false, interactive: false
            }).addTo(map);
          }
          if (window.location.hash) {
            viewFromHash();
          } else {
            map.fitBounds({{ this.bounds }}, {animate: false});
          }
          window.addEventListener("hashchange", viewFromHash);
        })();
        {% endmacro %}
        """
    )


LEGEND_HTML = """
<div style="position: fixed; bottom: 28px; left: 12px; z-index: 9999; background: white;
  padding: 10px 12px; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,.25);
  font: 13px/1.5 system-ui, sans-serif; color: #212529; max-width: 270px;">
  <div style="font-weight: 600; margin-bottom: 4px;">{title}</div>
  {rows}
  <div style="color: #6c757d; margin-top: 6px;">
    Coloured lines: the frozen graph used for evaluation. Base map: Esri, current data,
    for orientation only. More layers in the panel on the right.
  </div>
</div>
"""
LEGEND_ROW = (
    '<div><span style="display: inline-block; width: 26px; border-top: 4px {style} {colour};'
    ' vertical-align: middle; margin-right: 6px;"></span>{label}</div>'
)


def legend_html(title: str) -> str:
    """Return the fixed legend box of the map."""
    rows = [
        ("solid", COLOURS["two_way"], "Street, two-way"),
        ("solid", COLOURS["one_way"], "One-way segment, arrow = legal direction"),
        ("solid", COLOURS["alley"], "Alley"),
        ("dashed", COLOURS["internal"], "Internal road"),
        ("solid", COLOURS["unnamed"], "Unnamed edge (roundabout, slip lane, link)"),
        ("solid", COLOURS["askable"], "Street askable in Phase 1"),
        ("dashed", COLOURS["area"], "Study area (rectangle downloaded from OSM)"),
    ]
    return LEGEND_HTML.format(
        title=title,
        rows="".join(
            LEGEND_ROW.format(style=style, colour=colour, label=label)
            for style, colour, label in rows
        ),
    )


@dataclass
class MergedFeature:
    """Segments drawn as one multi-line feature: same layer, streets, direction and width.

    Leaflet re-projects, redraws and hit-tests every feature on each zoom and mouse
    move, so merging the about 3,000 segments into a few hundred features is what keeps
    the map responsive. The colours, widths and arrows are unchanged.
    """

    properties: dict[str, object]
    lines: list[list[tuple[float, float]]]
    length_m: int = 0


def add_to_merged(
    merged: dict[tuple[object, ...], MergedFeature],
    key: tuple[object, ...],
    points: list[tuple[float, float]],
    properties: dict[str, object],
    length_m: int,
) -> None:
    """Add one polyline to the merged feature of its key."""
    if key not in merged:
        merged[key] = MergedFeature(properties=dict(properties), lines=[])
    merged[key].lines.append(points)
    merged[key].length_m += length_m


def merged_features(merged: dict[tuple[object, ...], MergedFeature]) -> list[dict[str, object]]:
    """Return GeoJSON MultiLineString features with the summed length in the tooltip."""
    features: list[dict[str, object]] = []
    for item in merged.values():
        direction = "one-way" if item.properties.get("direction") == "one_way" else "two-way"
        properties = {**item.properties, "segment": f"{direction}, {item.length_m} m in total"}
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [[list(p) for p in line] for line in item.lines],
                },
                "properties": properties,
            }
        )
    return features


def point_feature(
    latitude: float, longitude: float, properties: dict[str, str]
) -> dict[str, object]:
    """Return a GeoJSON Point feature."""
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
        "properties": properties,
    }


def junction_layer(view: GraphView) -> folium.GeoJson:
    """Return a point layer of the junctions where regular streets meet."""
    features = []
    for junction in view.junctions.values():
        streets = sorted(
            view.streets[s].display_name for s in junction.street_ids if view.is_regular_street(s)
        )
        if len(streets) < 2 and not junction.place_names:
            continue
        features.append(
            point_feature(
                junction.latitude,
                junction.longitude,
                {
                    "junction": junction.junction_id,
                    "kind": junction.kind.value,
                    "nodes": f"{len(junction.node_ids)} nodes over {junction.spread_m:.0f} m",
                    "streets": ", ".join(streets),
                    "place": ", ".join(junction.place_names) or "-",
                },
            )
        )
    return folium.GeoJson(
        collection(features),
        name="Junctions (colour = kind)",
        show=False,
        marker=folium.CircleMarker(radius=5, fill=True, fill_opacity=0.9, weight=1),
        style_function=lambda feature: {
            "color": "#ffffff",
            "fillColor": JUNCTION_COLOURS[feature["properties"]["kind"]],
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["junction", "kind", "nodes", "streets", "place"],
            aliases=["Junction", "Kind", "Size", "Regular streets", "Place"],
            sticky=True,
        ),
    )


def flag_layer(flagged: Sequence[FlaggedItem]) -> folium.GeoJson:
    """Return a point layer that circles every flagged street and junction."""
    features = [
        point_feature(item.latitude, item.longitude, {"item": item.label, "flags": item.flags})
        for item in flagged
    ]
    return folium.GeoJson(
        collection(features),
        name="Review flags",
        show=False,
        marker=folium.CircleMarker(radius=14, fill=False, weight=3),
        style_function=lambda _feature: {"color": COLOURS["askable"]},
        tooltip=folium.GeoJsonTooltip(fields=["item", "flags"], aliases=["Item", "Flags"]),
    )


@dataclass(frozen=True)
class FlaggedItem:
    """A street or junction with review flags, placed on the map."""

    label: str
    latitude: float
    longitude: float
    flags: str


def build_map(
    view: GraphView,
    graph_version: str,
    study_area: BoundingBox,
    flagged: Sequence[FlaggedItem] = (),
) -> folium.Map:
    """Return the folium map of a reference graph."""
    bounds = [[study_area.south, study_area.west], [study_area.north, study_area.east]]
    centre = [(study_area.south + study_area.north) / 2, (study_area.west + study_area.east) / 2]
    world = folium.Map(
        location=centre,
        prefer_canvas=True,
        zoom_start=15,
        tiles=None,
        control_scale=False,
        zoom_snap=0.25,
        zoom_delta=0.5,
    )
    world.add_child(MetricScale())
    # Tile providers that need no API key (CartoDB tiles now require one).
    # Esri tiles need no API key and no HTTP Referer, so they also load when the page is
    # opened as a local file. OpenStreetMap's own tile servers block such requests, and
    # CartoDB tiles now need an API key.
    folium.TileLayer(
        "Esri.WorldGrayCanvas",
        name="Base map: light grey",
        max_native_zoom=16,
        max_zoom=19,
    ).add_to(world)
    folium.TileLayer(
        "Esri.WorldStreetMap",
        name="Base map: streets, faded",
        opacity=0.55,
        max_zoom=19,
        show=False,
    ).add_to(world)
    folium.TileLayer(
        "Esri.WorldImagery", name="Base map: satellite", max_zoom=19, show=False
    ).add_to(world)

    merged: dict[str, dict[tuple[object, ...], MergedFeature]] = {
        name: {} for name in ("street", "alley", "internal", "unnamed", "askable", "arrows")
    }
    label_lines: list[tuple[str, list[list[float]]]] = []
    for segment in map_segments(view):
        properties = segment_properties(view, segment)
        layer = layer_of(view, segment)
        streets_key = tuple(segment.street_ids)
        add_to_merged(
            merged[layer],
            (streets_key, segment.one_way, segment.width_m),
            segment.points,
            properties,
            segment.length_m,
        )
        if properties["askable"] == "yes":
            add_to_merged(
                merged["askable"],
                (streets_key,),
                segment.points,
                {**properties, "width_m": segment.width_m + 4},
                segment.length_m,
            )
        if layer == "street" and segment.length_m >= ARROW_MINIMUM_SEGMENT_M:
            label_lines.append(
                (
                    str(properties["name"]),
                    [[y, x] for x, y in reading_direction(segment.points)],
                )
            )
        # Unnamed one-way edges (roundabouts, slip lanes) get arrows too, whatever their length,
        # because they decide how one gets from one street to another.
        if segment.one_way and (
            (layer == "street" and segment.length_m >= ARROW_MINIMUM_SEGMENT_M)
            or layer == "unnamed"
        ):
            add_to_merged(
                merged["arrows"],
                (layer == "unnamed",),
                arrow_points(segment.points, segment.width_m),
                {"name": UNNAMED_LABEL if layer == "unnamed" else "arrow"},
                0,
            )
    layers = {name: merged_features(items) for name, items in merged.items()}

    folium.Rectangle(
        bounds=bounds,
        color=COLOURS["area"],
        weight=1.5,
        dash_array="6 6",
        fill=False,
        # On a canvas Leaflet hit-tests the whole inside of a polygon even without fill,
        # so an interactive rectangle would catch every mouse move over the map.
        interactive=False,
        name="Study area (download rectangle)",
    ).add_to(world)
    unnamed = plain_layer(layers["unnamed"], "Unnamed edges", COLOURS["unnamed"], True)
    unnamed.add_to(world)
    alleys = plain_layer(layers["alley"], "Alleys", COLOURS["alley"], False)
    alleys.add_to(world)
    internal = plain_layer(layers["internal"], "Internal roads", COLOURS["internal"], False, "6 4")
    internal.add_to(world)
    streets = folium.GeoJson(
        collection(layers["street"]),
        name="Streets (blue two-way, orange one-way)",
        style_function=street_style,
        highlight_function=lambda _feature: {"color": COLOURS["askable"], "opacity": 1},
        tooltip=tooltip(),
    )
    streets.add_to(world)
    folium.GeoJson(
        collection(layers["arrows"]),
        name="One-way arrows",
        style_function=arrow_style,
        interactive=False,
    ).add_to(world)
    askable = plain_layer(
        layers["askable"], "Streets askable in Phase 1", COLOURS["askable"], False
    )
    askable.add_to(world)

    Search(  # type: ignore[no-untyped-call]
        layer=streets,
        search_label="name",
        placeholder="Find a street, e.g. Pasteur",
        collapsed=False,
        search_zoom=17,
        position="topleft",
    ).add_to(world)
    Fullscreen(position="topleft").add_to(world)  # type: ignore[no-untyped-call]
    junction_layer(view).add_to(world)
    if flagged:
        flag_layer(flagged).add_to(world)
    layer_control = folium.LayerControl(collapsed=False)
    layer_control.add_to(world)
    world.add_child(
        ZoomStyling(
            [(unnamed, 1.5), (alleys, 1.2), (internal, 1.2), (streets, 2.0), (askable, 3.0)],
            label_lines,
            layer_control,
            bounds,
        )
    )
    root = world.get_root()
    if not isinstance(root, Figure):
        raise TypeError("A folium map is expected to have a Figure as its root.")
    root.html.add_child(folium.Element(legend_html(graph_version)))
    return world
