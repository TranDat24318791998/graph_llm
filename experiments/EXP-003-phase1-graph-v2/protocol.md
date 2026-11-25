# EXP-003: Phase 1 main comparison on reference graph v2

| | |
|---|---|
| **Status** | approved |
| **Research question** | RQ1, with first evidence for RQ2 (error stages), RQ3 (direction loss in the LLM-built graph) and RQ4 (easy method candidates M3N, M3T) |
| **Approved by owner on** | 2026-11-25. The owner delegated all Phase 1 decisions overnight and decided on 2026-11-25 that all Phase 1 experiments are rerun on graph v2. This protocol was written before any EXP-003 run. |

## 1. Question

How do accuracy and cost differ between representations of the same verified road graph, per question type? The representations are no retrieval (M0), text RAG (M1), Graph RAG with an LLM-built graph (M2, LightRAG), the OSM subgraph with local node ids (M3), the same subgraph with intersection names (M3N), and the M1 text selected through the street graph (M3T).

EXP-001 and EXP-002 asked the same on graph v1 (`hcmc_district1_v2_p1`). Their runs are pipeline trials, because graph v1 had modelling gaps that changed answers:
- divided roads;
- junctions made of several nodes;
- OSM ways with different names merged into one edge;
- turn restrictions ignored.

Hypotheses H1 to H7 below are carried over from those protocols. The trial results were seen before this protocol was written. The test split is not used.

## 2. Hypotheses

- **Hypothesis** H1: M3 is more accurate than M1 and M2 on L2 and L3.
- **Hypothesis** H2: M2 is not more accurate than M1 on L3, and breaks direction or turn rules on L4. LightRAG 1.5.8 stores an undirected graph.
- **Hypothesis** H3: every representation is weak on L4 and L5.
- **Hypothesis** H4: M0 is near the constant-guess baseline on every type.
- **Hypothesis** H5: M3N is more accurate than M3 on L2 and L3, and gives no node-id answers. Node-id answers in at most 5% of questions count as none.
- **Hypothesis** H6: M3T is at least as accurate as M1 on every type, and more accurate on L5.

A hypothesis is rejected when the paired exact McNemar test (p < 0.05) or the 95% Wilson intervals contradict it. With 50 questions per type, only large differences (about 20 points or more) can be shown.

## 3. Design

- **Independent variable:** representation, at six levels: M0, M1, M2, M3, M3N, M3T (definitions in `experiments/EXP-001-phase1-pilot/protocol.md` and `experiments/EXP-002-phase1-graph-context/protocol.md`).
- **Controlled:**
  - answering LLM `qwen2.5-7b-ctx16k` (Ollama), temperature 0, seed 0;
  - prompt `configs/prompts/answer_v1.yaml`;
  - JSON answer formats;
  - context budget of 16,000 characters;
  - the question linker links regular streets only, for every representation that links (M3, M3N, M3T);
  - M1 top-30;
  - M3 and M3N 2 hops;
  - M3T 2 street hops;
  - M2 with the same LightRAG settings as EXP-001.
- **Data:**
  - graph `hcmc_district1_v3_p2` (frozen 2026-11-25, reviewed: `configs/graphs/hcmc_district1_v3_p2_review.yaml`);
  - benchmark `v0.3`: 250 development questions, 50 per type, built from `configs/benchmark/v0_3.yaml`;
  - text corpus `hcmc_district1_v3_p2_text_v2`, template text generated from graph v2;
  - LightRAG index built from that corpus with `qwen2.5:14b`, 3,000-character packs as in EXP-001.
- **Answer definitions** (graph v2):
  - Two streets meet when they reach the same junction.
  - A street is one-way only if it is one-way along its whole length; divided roads are two-way.
  - Routes obey edge directions and unconditional car turn restrictions.
  - Route questions whose answer passes a time-conditional restriction are excluded.
  - Streets excluded by the graph review are never asked about.

## 4. Implementation

- Config: `configs/experiments/exp003.yaml`.
- Order, one at a time:
  1. build the corpus and the LightRAG index;
  2. run M0, M1, M3, M3N, M3T;
  3. run M2.
- Commands:
  - `uv run python scripts/build_text_corpus.py hcmc_district1_v3_p2 configs/llm/ollama_qwen2.5_7b.yaml --text-version v2`
  - `uv run python scripts/build_lightrag_index.py hcmc_district1_v3_p2_text_v2 configs/llm/ollama_qwen2.5_14b.yaml`
  - `uv run python scripts/run_experiment.py configs/experiments/exp003.yaml <representation>`
- Code version: every run manifest records the commit (marked `-dirty`) and a SHA-256 of `src/`.

## 5. Evaluation

The metrics are those of EXP-001 Section 5, with two changes:
- The route sub-score `wrong_direction` is renamed `breaks_direction_or_turn_rule`: the predicted streets can be followed in order, but only against an edge direction or through a forbidden turn.
- Legality and optimality of a predicted route are checked with turn restrictions.

Additional counts:
- node-id answers (`"n<digits>"`) per run;
- evidence recall per representation, measured as in EXP-001. The known limitation for text contexts applies: a street named in another street's sentence is not counted as present.

## 6. Expected cost

- Index: about 4 hours with `qwen2.5:14b`.
- Answering: six runs of 250 questions, about 4 to 5 hours in total, with M2 the slowest.

## 7. Deviations



- 2026-11-25, before any EXP-003 run: run order changed to M0, M1, M3, M3N, M3T first, then the LightRAG index on corpus text_v2, then M2, so that the five runs that need no index finish overnight. Runs still execute one at a time; the only earlier job sharing the machine is the EXP-001 M2 reference run, which finishes before the first EXP-003 run starts.
- 2026-11-25, before any EXP-003 run: the corpus is `hcmc_district1_v3_p2_text_v2` (the first build, text_v1, described places as streets and was replaced before use; see `PROGRESS.md`).
