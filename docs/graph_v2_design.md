# Reference graph v2: processing design and review protocol

**Status:** open questions decided by the owner on 2026-11-25 (Q-A: option a; Q-B: option ii; Q-C: regular streets only); implementation in progress.
**Scope:** how OpenStreetMap data is turned into the reference road graph used for every answer, every piece of evidence and every representation. This replaces the street-level part of graph v1 (`hcmc_district1_v2_p1`); the downloaded snapshot `hcmc_district1_v2` stays as it is.

## 1. Why a v2

The reference graph is one of the thesis contributions: a road-network graph that is large, verified, and correct with respect to OSM, so that question answers computed from it can be trusted. Graph v1 kept the OSM edges exact, but its street-level facts were defined too simply. Reviewing the v1 map with the owner surfaced three gaps:

1. **Divided roads.** OSM stores a divided road (two carriageways separated by a median) as two parallel one-way ways. v1 put such a street, for example Nguyễn Huệ, Lê Lợi or Tôn Đức Thắng, in the same "mixed" class as a street that really changes from one-way to two-way. 94 of 298 regular streets (32%) fell in that class, including most main roads of District 1. As a result, attribute questions never asked about them, and the count "how many neighbouring streets are two-way" treated divided roads as not two-way (17 of 25 such questions involve one).
2. **Junctions made of several nodes.** One real intersection is often several OSM nodes: a roundabout ring (10 rings in the area), the two crossings of a divided road, slip lanes. v1 says two streets meet only when they share a node, so streets that meet at a roundabout or across a divided road can be missed (31 pairs of regular streets connected only through unnamed edges; for example Phạm Ngọc Thạch and Lê Duẩn).
3. **Named places that are not streets.** Squares and roundabouts carry street-like names (Công trường Quốc Tế, Ngã sáu Phù Đổng, Công trường Dân Chủ); v1 treated them as streets.

Measured on the v1 road layer (1,976 nodes, 4,370 directed edges):

| Quantity | Value |
|---|---|
| Edges shorter than 10 m / 20 m | 412 / 866 |
| Groups of nodes joined by edges shorter than 25 m | 329 (up to 15 nodes) |
| Roundabout rings (`junction=roundabout`) | 10, of which 5 named |
| Unnamed edges | 640: 306 residential, 289 `*_link`, 37 primary/secondary/tertiary, 8 living street |
| One-way length on regular streets that pairs with an opposite carriageway of the same street within 40 m | 53.3 of 124.7 km, on 64 streets |
| Dead-end nodes | 277: 104 alleys, 96 unnamed, 41 regular streets, 33 at the area boundary |
| Bridges / tunnels (edges) | 42 / 2 |

## 2. Principles

1. **OSM is the ground truth for facts.** Geometry, direction (`oneway`), names, lanes and road class come from OSM and are never edited. If OSM looks wrong compared with reality, the case is recorded as an OSM discrepancy, and questions avoid that street or junction. The OSM data itself is not corrected (see open question Q-A).
2. **Interpretation must match common understanding.** Rules decide what OSM facts mean at the level people ask about: what counts as a street, a junction, a one-way street, a street's length. Every rule is written down here, implemented once, and unit-tested on small hand-built cases that mirror the real situations (roundabout, divided-road crossing, slip lane, overpass).
3. **Rules first, overrides last.** Automatic rules handle the general case. An override is used only for a case the rules cannot decide, carries a written reason, and is counted. The number of overrides per kilometre of road is reported. A rising count on a larger area means the rules need work.
4. **Never merge by distance alone.** Nodes are grouped only along short connecting edges, never because they are close. Otherwise a bridge passing over a street would wrongly become an intersection.
5. **Routing stays on the exact OSM edges.** Junctions and street segments are layers on top of the road layer. Shortest routes and legality checks keep using every directed OSM edge unchanged.

## 3. Model

### 3.1 Road layer (unchanged from v1)

The snapshot reduced to its largest strongly connected component. Edges are exact OSM edges after OSMnx simplification.

### 3.2 Junction layer (new)

A **junction** is a set of road-layer nodes that together form one real intersection. A junction is built from:

- **roundabout rings**: all nodes of a connected set of `junction=roundabout` edges;
- **junction-internal edges**: short edges that only connect parts of one intersection. These are:
  - unnamed `*_link` edges and unnamed edges up to `link_max_m`;
  - the short piece of a crossing street between the two carriageways of a divided road, up to `median_crossing_max_m`;
  - a named edge up to `tiny_edge_max_m` between two intersection nodes;
- every other node where at least two named streets meet, or at least three edges meet, as a junction of its own.

Junctions are the connected components of the junction-internal edges, together with the rings. Each junction records:
- its member nodes and centroid;
- the streets that reach it;
- its kind: simple, divided crossing, roundabout, or complex;
- its name, when a named place such as Công trường Quốc Tế forms its ring.

Values used (`configs/graphs/hcmc_district1_v3_p2.yaml`): `link_max_m` 80, `median_crossing_max_m` 40, `tiny_edge_max_m` 12. An edge that carries a place name and another street name is internal only if it is at most `link_max_m` long.

**Two streets meet** when they reach the same junction. This replaces "share a node" for L2, L3, L5, the text corpus and the linker's notion of "intersection of A and B".

### 3.3 Street layer with segments (extended)

A street (v1 rules for names, merging and categories are kept) is split into **street segments** between consecutive junctions. Each segment is one of:

- **two-way**: a single carriageway with edges in both directions;
- **divided**: two carriageways of the same street, each one-way, in opposite directions, paired between the same two junctions; the centre-line length is the mean of the two carriageway lengths;
- **one-way**: a single carriageway with edges in one direction only.

From its segments, a street gets a **direction status**:

| Status | Meaning |
|---|---|
| `two_way` | every segment two-way |
| `two_way_divided` | every segment two-way or divided, at least one divided (counts as two-way for questions) |
| `one_way` | every segment one-way, all in one consistent direction along the street |
| `mixed` | anything else; its segment breakdown is stored and described |

A street's **centre-line length** is the sum of its segment centre-line lengths.

### 3.4 Places (new)

A named roundabout or square is a **place**, not a street: it names a junction. Attribute questions (one-way, length) are not asked about places. Its ring edges keep their name for display and text, and the streets around it meet each other at that junction.

### 3.5 Corpus and questions

The text corpus describes streets segment by segment, using junction names. For example: "Đoạn Nguyễn Huệ từ giao lộ Lê Lợi đến giao lộ Lê Thánh Tôn: 140 m, hai chiều có dải phân cách." Question templates use the new street statuses and junctions. A question is asked about a street or junction only if its review status is `accepted`.

## 3.6 Implementation notes (2026-11-25)

- **Unsimplified source.** OSMnx's default simplification merges consecutive OSM ways even when their names differ (45 such name combinations in snapshot `hcmc_district1_v2`, for example a bridge merged into the street it leads into), and the merged edge cannot be split again. Graph v2 therefore starts from the unsimplified snapshot `hcmc_district1_v3` and simplifies it itself (`graph/simplify.py`): a node is kept wherever the incident edges differ in name, highway, oneway, junction, bridge, tunnel, lanes, maxspeed or access, and wherever a turn restriction acts.
- **Carriageway pairing** is per (street, edge): an edge with two names is classified within each street. Opposite one-way edges that share a node (carriageways splitting or rejoining) pair only if they are at least 4 m apart at the midpoint; closer pairs are hairpins.
- **Turn restrictions** are frozen next to the snapshot (`data/raw/<area_id>_restrictions/`) and mapped to forbidden edge sequences, including via-way restrictions (U-turns across a median). Each is mapped, already impossible, outside the graph, or unmappable (none).
- **Code:** `graph/carriageways.py`, `graph/junctions.py`, `graph/restrictions.py`, `graph/simplify.py`, `graph/reference_graph.py`, `graph/review.py`, `questions/graph_view.py`, `questions/routes.py`, `questions/turns.py`; build with `scripts/build_reference_graph.py`.

## 4. Review protocol

Goal: every regular street and every junction is checked once, the check is recorded, and the owner can audit it.

1. **Automatic checks** produce flags. Examples:
   - a street whose segments cannot be classified;
   - a divided segment whose carriageways differ in length by more than 30%;
   - a junction with more than `n` nodes or wider than 80 m;
   - a regular-street dead end that is not at the boundary;
   - a named edge shorter than 5 m;
   - two streets that meet in v2 but not in v1, or the reverse;
   - a one-way street whose direction flips along its length;
   - a name used by streets far apart;
   - an edge carrying several names.
2. **Street cards and junction cards.** A generated review report (`docs/reports/graph_review_<version>.md`) has one card per regular street and one per junction. A card lists the computed facts, the flags, and a link to the location in the interactive map, where flagged items are highlighted.
3. **Review decisions.** Every card is reviewed by Claude against the OSM data and the base map/satellite layer. The result is recorded in a review file in the graph config: `accepted`, `accepted with override` (override and reason), or `excluded from questions` (reason, for example an OSM discrepancy). A card whose facts or flags change in a later build needs review again.
4. **Owner audit.** The owner checks every flagged card and a random sample of unflagged ones, for example 30 streets and 30 junctions, on the interactive map using local knowledge. Findings go back as rule changes, or as overrides with reasons.
5. **Acceptance.** Graph v2 is frozen when:
   - no flag is unresolved;
   - every override has a reason;
   - the owner's sample has no unexplained error;
   - all rule unit tests pass.

   The build report states the counts: streets and junctions reviewed, flags by type, overrides per km, items excluded.

## 5. Scaling

The same rules and review report run on any area. For a larger area, all cards are still generated, but full manual review is not feasible. Instead:
- flagged cards are always reviewed;
- unflagged cards are sampled, and the error rate in the sample is reported;
- the override count per km shows whether the rules generalize.

This is why the rules, flags and tests matter more than the individual fixes in District 1.

## 6. Questions decided by the owner (2026-11-25)

- **Q-A, OSM discrepancies. Decided: (a).** When OSM disagrees with reality, for example a one-way tag missing on a street the owner knows is one-way, should we (a) keep OSM as ground truth and exclude the item from questions (current principle), or (b) apply a recorded correction?
  - (a) keeps "correct with respect to OSM" clean and reproducible.
  - (b) makes the graph closer to reality, but it would no longer be pure OSM.
- **Q-B, turn restrictions. Decided: unconditional car restrictions enforced; time-conditional ones handled by option (ii).** OSMnx ignores OSM turn restrictions. They were measured with an Overpass query on the same rectangle on 2026-11-25. The Overpass data was dated 2026-07-24, so the counts are indicative; the real data would be frozen with the graph.
  - There are 391 restriction relations. 107 always apply: 51 `no_left_turn`, 28 `no_right_turn`, 24 `no_u_turn`, 4 `no_straight_on`. 283 apply only at certain hours (`restriction:conditional`, mostly 06:00-22:00 or rush hours).
  - 334 restrictions exempt motorcycles and bicycles, so they target cars. Our route questions ask about cars.
  - Ignoring them would make many "shortest legal route for a car" answers illegal in reality, so they cannot be declared out of scope.
  - Proposal:
    - download the restriction relations once and freeze them with the snapshot (recording the download date);
    - apply the unconditional car restrictions in routing, through a turn-aware (edge-based) shortest path;
    - treat time-conditional restrictions with one of two options: (i) fix a stated time of day for all route questions, or (ii) exclude route questions whose answer passes a time-conditional restriction.
- **Q-C, alleys. Decided: only junctions where regular streets meet are reviewed.** Alleys stay out of Phase 1 questions. Should alley junctions still be reviewed, or only the junctions where regular streets meet?

## 7. Work plan

1. Measure turn restrictions in the area (done: 391 relations, 107 unconditional; see Q-B). Freeze the restriction relations next to the snapshot.
2. Implement the junction layer, then the divided-segment pairing and the street segments, each with unit tests on hand-built cases.
3. Implement the automatic checks, the review report and the map layers (junctions, segment kinds, flags).
4. Review every street and junction card; record the decisions.
5. Owner audit of flagged and sampled cards; adjust the rules.
6. Freeze graph v2 (`hcmc_district1_v2_p2`). Regenerate the corpus, the LightRAG index and the benchmark (v0.3). Rerun the Phase 1 experiments under a revised protocol.
