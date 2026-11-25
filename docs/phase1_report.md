# Phase 1 Report: a first end-to-end comparison of graph representations for road-network QA

**Status:** draft, results pending for M2 (EXP-001) and EXP-002.
**Date:** 2026-11-24
**Scope:** Phase 1 as set by the owner: (1) a normalized, frozen road graph; (2) a 250-question evaluation set, 50 per type; (3) Graph RAG (LightRAG) and one easy-to-implement method; (4) comparison and analysis; (5) this report.

All numbers below are generated from run directories by `scripts/analyze_experiment.py`; the full tables are in `experiments/EXP-001-phase1-pilot/results/results.md` and `experiments/EXP-002-phase1-graph-context/results/results.md`.

## 1. Summary

PENDING (written last).

## 2. The road graph

**Source.** OpenStreetMap drive network of a fixed rectangle around District 1 of Ho Chi Minh City (west 106.680, south 10.755, east 106.713, north 10.797), downloaded with OSMnx 2.1.2 on 2026-11-24 as snapshot `hcmc_district1_v2` (2,026 nodes, 4,451 directed edges, 355.5 km; GraphML SHA-256 `94e84d85...`). The snapshot is frozen and checksummed.

**Attribute coverage** (`docs/reports/osm_coverage_hcmc_district1_v2.md`). `name` covers 85.5% of edges (95.0% of length), `lanes` 50.3%, `maxspeed` 22.3%. `oneway` and `length` are always present because OSMnx fills them in. Decision: missing attributes are not imputed and not taken from other sources such as Google Maps, because ground truth is defined as "correct with respect to OSM"; questions are only asked about attributes that are present.

**Processed graph `hcmc_district1_v2_p1`** (`docs/reports/street_model_hcmc_district1_v2_p1.md`), in two layers:

- *Road layer*: the snapshot reduced to its largest strongly connected component, so every intersection can reach every other one legally. This removes 50 nodes and 81 edges at the edges of the rectangle; no edge is added or changed. Result: 1,976 nodes, 4,370 directed edges, 640 of them unnamed.
- *Street layer*: named edges grouped into 520 streets (298 regular streets, 205 alleys, 17 internal roads). Pieces with the same normalized name are joined when they are at most 200 m apart along the network. The 19 names still split after that rule were reviewed one by one: 9 were merged (pieces of one street interrupted by a roundabout, a bridge ramp or a short unnamed link) and 10 were kept apart (different streets sharing a name, 1.7 to 5.8 km apart, for example the two "Lê Quý Đôn"). 278 regular streets have a name used by no other street; Phase 1 asks only about these.
- *Names*: Vietnamese names are normalized (Unicode NFC, the look-alike letter "Ð" (U+00D0) mapped to "Đ", leading "Đường" ignored for matching). An alias dictionary covers common short forms (CMT8, NKKN, NTMK, ĐBP, XVNT). Names without diacritics are used only for lookups, never to merge streets.

Decisions taken with the owner: no imputation; no questions about alleys or internal roads in Phase 1; bridges and squares count as ordinary streets; same-name pieces merged by the two-layer rule; only the largest strongly connected component kept.

## 3. The benchmark (v0.2)

250 questions in Vietnamese, 50 per type, all in the development split. Every answer and its required evidence (the set of OSM edges needed to answer) is computed by code from the processed graph. An LLM (`qwen2.5:7b`) only paraphrases the template text; a paraphrase is kept only if every street name and every required phrase (for example "một chiều", "tuân thủ") survives, otherwise the template text is used.

| Type | Template | n | What is asked | Answer |
|---|---|---|---|---|
| L1 attribute | `L1_oneway` | 25 | Is street S one-way? (balanced yes/no) | boolean |
| | `L1_length` | 25 | Is the total length of S greater than N m? (balanced; N at least 25% away from the true value) | boolean |
| L2 adjacency | `L2_neighbours` | 50 | Which regular streets meet S? (2 to 8 streets) | street set |
| L3 multi-step directional | `L3_next_on_oneway` | 50 | At the junction of X and one-way street Y, following Y in its legal direction, which street comes next? | street name |
| L4 shortest legal route | `L4_shortest_route` | 50 | Shortest car route between two intersections, respecting one-way rules (400 to 2,000 m, 2 to 5 streets) | street sequence and length |
| L5 aggregation | `L5_two_way_neighbours` | 25 | Among the streets meeting S, how many are two-way along their whole length? | number |
| | `L5_common_neighbours` | 25 | How many streets meet both S and T? | number |

Count answers are stratified by value so that a constant guess is weak (the constant-guess baseline is reported with the results). 194 of 250 questions have an accepted paraphrase that passed the check, and 171 of these differ from the template text. The 56 fallbacks are 45 L4 questions (the paraphraser dropped "tuân thủ") and 11 L3 questions.

A first draft (v0.1) was discarded before any run because lane-count answers were almost always "2", L5 counts were skewed, and one paraphrase changed "respect one-way rules" into "follow one-way streets" without failing the check.

## 4. Methods

All representations use the same answering LLM, `qwen2.5:7b` (Q4_K_M, Ollama, 16k context, temperature 0, seed 0), the same answering prompt (`configs/prompts/answer_v1.yaml`), JSON answers, and the same context budget of 16,000 characters. Only the context differs.

| ID | Representation | Context |
|---|---|---|
| M0 | No retrieval | empty |
| M1 | Text RAG | top-30 chunks by bge-m3 cosine similarity from 520 template-generated street descriptions (631 chunks), added in rank order while they fit |
| M2 | Graph RAG, LightRAG 1.5.8 | `mix` mode context (knowledge-graph entities and relations plus text chunks) over a graph that `qwen2.5:14b` extracted from the M1 text; the OSM graph is withheld |
| M3 | OSM subgraph | 2-hop neighbourhood of the streets named in the question, one segment per line with direction arrows (`n1 -> n2` one-way, `n1 <-> n2` two-way), length and lanes |
| M3N | OSM subgraph, named nodes (EXP-002) | as M3, but nodes are written as the streets meeting there (`[Lê Lợi / Pasteur]`) |
| M3T | Graph-linked text (EXP-002) | the M1 street descriptions, selected through the street graph: the linked streets, then the streets meeting them, then one street further |

**Text corpus.** Each street description is generated from the graph by fixed sentence templates: a header (category, one-way status, lane count when known, the streets it meets) and one sentence per segment between intersections, for example "Đoạn Pasteur từ giao lộ Lê Lợi đến giao lộ Lê Thánh Tôn: 120 m, một chiều (xe chỉ được đi từ giao lộ Lê Lợi về giao lộ Lê Thánh Tôn)." No LLM rewrites this text, so it contains only facts of the graph.

**LightRAG.** In version 1.5.8 the knowledge graph is an undirected `networkx.Graph` and the extraction prompt asks for undirected relations; one-way information can only survive in description text. To make indexing feasible on the local machine, the 520 descriptions were joined into 145 packs of at most 3,000 characters (one extraction call per pack instead of per document; recorded as a protocol deviation before any M2 run).

**Scoring** (no LLM judge). L1, L3, L5: exact match (street names compared after normalization, a match without diacritics accepted). L2: set F1; accuracy counts F1 = 1. L4: correct if a legal route exists that follows the predicted street sequence in order and is at most 2% longer than the true shortest route; legality, wrong one-way direction and length within 10% are also reported.

**Diagnostics.** Evidence recall is the share of required evidence edges present in the context (M1 and M3T: the edges described by the included chunks; M2: edges whose corpus sentence appears verbatim in the context; M3, M3N: the serialized edges). Each wrong answer is assigned to the first failing stage: output format, no context (M0), entity linking, retrieval (evidence recall below 1), or usage (all evidence present but answer wrong). Accuracy has 95% Wilson intervals; representations are compared on the same questions with exact McNemar tests (exploratory, no multiple-comparison correction).

## 5. Results

### 5.1 EXP-001: M0, M1, M2, M3

PENDING M2. Results without M2:

| Type | Constant guess | M0 | M1 | M3 |
|---|---|---|---|---|
| L1 | 52.0 | 52.0 [38.5, 65.2] | 84.0 [71.5, 91.7] | 74.0 [60.4, 84.1] |
| L2 (accuracy / mean F1) | 4.0 | 0.0 / 3.2 | 62.0 / 92.6 | 4.0 / 44.1 |
| L3 | 4.0 | 0.0 | 12.0 [5.6, 23.8] | 0.0 |
| L4 | 2.0 | 0.0 | 2.0 | 0.0 |
| L5 | 24.0 | 12.0 | 20.0 | 16.0 |
| All | 17.2 | 12.8 | 36.0 [30.3, 42.1] | 18.8 [14.4, 24.1] |

PENDING: interpretation, error stages, cost, M2 index cost and alignment.

### 5.2 EXP-002: presenting graph evidence

PENDING.

## 6. Discussion

PENDING.

## 7. Limitations

PENDING.

## 8. Implications for Phase 2 and RQ4

PENDING.
