# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This is a master's thesis project at an early stage. The project skeleton and the core schemas exist; the graph pipeline, question generation, representations M0-M4 and evaluation are not implemented yet.

- `docs/thesis_proposal.md` is the source of truth for the research design. Read it before designing or implementing anything.
- `plan/PLAN.md` tracks progress: milestone status, task checklists, open questions and the decision log. Check it at the start of a session to find the current focus, and update it when a task is finished, a decision is made, or a question is resolved.
- `docs/rq4_method_directions_survey.md` surveys candidate RQ4 method directions; read it before work that touches RQ4.

## How the existing code fits together

- **Study area → snapshot.** A YAML file in `configs/areas/` is validated into `StudyArea` (bounding box, `network_type`, `simplify`). `graph/snapshot.py` downloads it once with OSMnx into `data/raw/<area_id>/graph.graphml` and writes a `SnapshotMetadata` record (`metadata.json`, including the GraphML SHA-256). `load_snapshot` returns `(RoadGraph, SnapshotMetadata)` after checking the checksum.
- **Coverage survey.** `graph/coverage.py` measures, for each OSM edge attribute, how many edges have it and what share of total length they cover. It produces a `CoverageReport` that `scripts/survey_coverage.py` renders to `docs/reports/`. Use `has_value` / `as_list` from this module when reading edge attributes that may be missing, NaN, or lists.
- **Reference graph v2** (current; design and review protocol in `docs/graph_v2_design.md`, built by `graph/reference_graph.py`). Source: an unsimplified snapshot plus its frozen turn restrictions (`graph/restrictions.py`), simplified without merging different roads (`graph/simplify.py`). On top of the road layer: carriageway labels per (street, edge) (`graph/carriageways.py`: two-way, divided, one-way; street direction `two_way`/`two_way_divided`/`one_way`/`mixed`; centre-line length), junctions (`graph/junctions.py`: groups of nodes joined by roundabout, place, connector, median-crossing or tiny edges; two streets meet when they reach the same junction), places (named squares and roundabouts are junction names, not streets). Every regular street and junction gets a review card (`graph/review.py`); decisions live in `configs/graphs/<version>_review.yaml`. Routing obeys unconditional car turn restrictions (`questions/turns.py`). `GraphView` is loaded with `load_graph_view()`.
- **Processed graph v1** (legacy, frozen `hcmc_district1_v2_p1`; `graph/processed.py`, `graph/streets.py`, config in `configs/graphs/`). Two layers. The road layer is the snapshot reduced to its largest strongly connected component, with no edge added or changed; all routing and answer computation runs on it. The street layer (`Street` records in `streets.json`) groups named edges into streets: same-name pieces are joined when at most `merge_gap_m` apart along the network, and any name still in several groups must have a reviewed `merge`/`separate` override, otherwise the build fails. Streets have a category (regular, alley, internal) and `shares_name_with_other_street`; Phase 1 questions use only regular streets with an unambiguous name. A processed graph version is frozen like a snapshot: a config change needs a new `graph_version`.
- **Name normalization** (`names/normalize.py`). `name_key` decides whether two spellings are the same name; `ascii_key` is for lookups only and must never be used to merge names.
- **Questions** (`questions/`). `GraphView` defines every street-level fact once (one-way status, centre-line length, meeting streets, clean intersections); `generate.py` lists all valid candidates per template with answer, evidence and features, then samples with a fixed seed (stratified where answers would be skewed). `routes.py` computes shortest legal routes and is reused to score predicted routes. Templates and paraphrase rules live in `configs/benchmark/`.
- **Representations** (`representations/`). M0 empty context; M1 bge-m3 retrieval over the template-generated corpus (`text_corpus.py`); M2 LightRAG 1.5.8 used only to build the context (its graph is undirected); M3 2-hop subgraph serialized with direction arrows (`subgraph.py`), seeded by `StreetLinker`. All share one answering prompt (`configs/prompts/answer_v1.yaml`) and one context budget.
- **Evaluation** (`evaluation/`). `runner.py` writes one run directory per representation; `scoring.py` is exact match / set F1 / graph-checked routes; error stages are output format, no context, entity linking, retrieval, usage. Tables come from `analysis/` via `scripts/analyze_experiment.py`.
- **LLM access** (`llm/client.py`). All chat calls go through `LLMClient`, which caches responses in `.cache/llm_cache.sqlite` keyed by the full request; Ollama context length is set by model variants (`qwen2.5-7b-ctx16k`, `qwen2.5-14b-ctx16k`) because the OpenAI-compatible endpoint ignores `num_ctx`.
- **Benchmark records** (`schemas/question.py`). A `Question` holds the template text and its paraphrase, the slot values, an `answer` (a discriminated union on `kind`: boolean, street_name, street_set, number, route), the required `Evidence` (node ids plus `EdgeRef(source, target, key)`, since the graph is a multigraph) and `QuestionFeatures` (hop count, direction dependence, entity-linking challenges). The features exist so that RQ4 routing can be developed later without regenerating the benchmark.
- **Run records** (`schemas/run.py`). A `RunManifest` holds one per run; a `Prediction` (exact context, raw response, parsed answer, token counts, LLM call count, latency) and a `Score` (`is_correct`, partial `score`, `evidence_recall`, `ErrorStage`) hold one per question. Together they give the per-question, per-representation comparability and cost logging that RQ4 depends on. File layout under `runs/` is in `experiments/README.md`.
- Tests never touch the network: coverage tests build small synthetic `MultiDiGraph`s, and snapshot tests use `tmp_path`. Keep it that way.

## Commands

The project uses Python 3.12 managed by `uv`.

```bash
uv sync                                   # create .venv and install dependencies
uv run pytest                             # run all tests
uv run pytest tests/test_schemas.py::test_unknown_field_is_rejected   # run one test
uv run ruff format .                      # format
uv run ruff check .                       # lint (Google-style docstrings are enforced)
uv run mypy                               # strict type check of src/ and tests/
uv run pre-commit install                 # once per clone: run the checks on every commit

uv run python scripts/download_graph.py configs/areas/hcmc_district1.yaml   # download and freeze an OSM snapshot
uv run python scripts/survey_coverage.py hcmc_district1_v2                  # regenerate the attribute-coverage report
uv run python scripts/build_graph.py configs/graphs/hcmc_district1_v2_p1.yaml  # build and freeze the processed graph
uv run python scripts/download_restrictions.py configs/areas/hcmc_district1_v3.yaml   # freeze OSM turn restrictions next to a snapshot
uv run python scripts/build_reference_graph.py configs/graphs/hcmc_district1_v3_p2.yaml [--dry-run]   # graph v2: build, review report, freeze
uv run python scripts/render_map.py configs/graphs/hcmc_district1_v3_p2.yaml   # folium web map with review flags -> docs/reports/map_<version>.html
uv run python scripts/build_benchmark.py configs/benchmark/v0_2.yaml configs/llm/ollama_qwen2.5_7b.yaml   # generate + paraphrase a benchmark version
uv run python scripts/build_text_corpus.py hcmc_district1_v2_p1 configs/llm/ollama_qwen2.5_7b.yaml       # street descriptions + bge-m3 embeddings (M1/M2 input)
uv run python scripts/build_lightrag_index.py hcmc_district1_v2_p1_text_v1 configs/llm/ollama_qwen2.5_14b.yaml  # M2 index (hours)
uv run python scripts/run_experiment.py configs/experiments/exp001.yaml M0 M1 M2 M3                     # one immutable run per representation in runs/
uv run python scripts/push_data.py                       # upload git-ignored data to gdrive:graph_llm_data, write data/data_manifest.json
uv run python scripts/setup_environment.py --verify-rebuild   # on another machine: download, verify checksums, rebuild check, tests
uv run python scripts/analyze_experiment.py configs/experiments/exp001.yaml experiments/EXP-001-phase1-pilot/results runs/<id> ...
```

Run scripts from the repository root; they use paths relative to it.

All three checks (ruff, mypy strict, pytest) must pass before work is considered done.

## Repository layout

```
src/graph_llm/     library code; all logic lives here
  schemas/         pydantic models for every record written to disk
configs/           one YAML file per run configuration, validated by pydantic
scripts/           thin entry points that only parse arguments and call the library
tests/
data/raw/          OSM snapshot, never modified (not in git)
data/processed/    cleaned graph and generated text (not in git)
data/benchmark/    versioned question sets (in git)
experiments/       one folder per experiment: approved protocol, results, notes
runs/              immutable raw outputs of every run (not in git)
docs/              proposal and thesis material
plan/              PLAN.md, the progress tracker
```

Conventions:

- Schemas subclass `StrictModel` (`schemas/base.py`): frozen and `extra="forbid"`, so a schema mismatch fails loudly. Changing a schema that already has data on disk requires a new benchmark or run version, not a silent edit.
- An OSM snapshot in `data/raw/<area_id>/` is frozen: `download_snapshot` refuses to overwrite it and `load_snapshot` verifies its checksum. OSM changes daily, so never re-download an existing area id; define a new area id instead. The snapshot is not in git, so it must be backed up separately.
- The road graph is an OSMnx `MultiDiGraph` (type alias `RoadGraph`). After simplification an edge attribute such as `name` or `highway` can be a list instead of a single value; code that reads attributes must handle both.
- Files in `docs/reports/` are generated by scripts and are not edited by hand.
- The experiment workflow (protocol first, immutable runs, results generated by script) is defined in `experiments/README.md`. Follow it exactly.
- Documents in the repo are written in English.
- **Two environments.** The primary environment is the owner's local Mac (Apple M4 Pro, 48 GB unified memory). The office server is secondary. Git-ignored data (`data/raw/`, `data/processed/`, `runs/`, `.cache/`) is shared through Google Drive with rclone (remote `gdrive:graph_llm_data`, rclone installed in `~/bin`): `scripts/push_data.py` uploads it (copy only, never deletes) and writes `data/data_manifest.json`, which is committed; `scripts/setup_environment.py` downloads it and verifies every file against that manifest (`src/graph_llm/data_sync.py`). Step-by-step instructions, including rclone installation and headless authorization on the server, are in `docs/data_sync.md`. The rclone config holds an OAuth token and is never committed. Data ids record where data came from: snapshot `hcmc_district1` exists only on the server, and `hcmc_district1_v2` is the local one.
- **LLM backends.** LLM code talks to an OpenAI-compatible endpoint whose base URL, model name and API key come from configuration, so the backend can be switched without code changes.
  - Local (current): Ollama at `http://127.0.0.1:11434/v1`. Pull a model only after checking free disk space and agreeing on the model with the owner.
  - Server (later): the already-running shared vLLM server at `http://127.0.0.1:8200/v1`, served model name `theracore-llm` (Qwen3-32B, fp8, 32768-token context), used as a client only. Its API key is the `--api-key` value on the running `vllm serve` command line; read it from the environment, do not commit it. The server is shared: never touch anything outside this project directory there (no other users' files, containers, databases, or vLLM servers), and check disk and GPU memory before loading anything.
  - The models for the thesis experiments are not chosen yet (open question Q2 in `plan/PLAN.md`).

## Working rules

These were set by the project owner and apply to all work in this repository.

- **English only in code.** Code, comments, docstrings, identifiers, and commit messages are written in English. No emoji or icons anywhere in code, comments, logs, or docs. Vietnamese appears only in data (questions, street names) and in conversation with the owner.
- **Clarity over cleverness.** Prefer explicit, readable code: descriptive names, small functions, typed signatures, no implicit behaviour or dense one-liners.
- **Explicit schemas and models.** Every data structure that crosses a module boundary or is written to disk has a defined, typed schema.
- **No experiment without prior agreement.** An experiment is one complete experimental design that produces a complete set of results. Writing code, running tests and trying things out while developing are normal work and need no approval. Before running any experiment, agree with the owner on three things: what the experiment is, how it will be implemented, and how it will be evaluated. Do not launch runs, change an agreed protocol, or alter metrics on your own initiative. This is a thesis, so experiments must be clean and defensible.
- **Subagents run on Opus 5.5, never Fable.** When spawning a subagent, always pass `model: "opus"`. Do not use `subagent_type: "fork"`, because a fork inherits the parent model and ignores the override.

## What the project is

A controlled comparison of how graph-structured knowledge should be presented to an LLM for question answering. The knowledge is held fixed (the Ho Chi Minh City road network from OpenStreetMap, loaded with OSMnx as a directed graph) and only the representation varies. Questions are in Vietnamese.

**The thesis must contribute a solution, not only a comparison.** RQ1-RQ3 are diagnostic (which representation works where, where errors come from, how faithful an LLM-built graph is). RQ4 is the required method contribution: use those findings to design a method that is more accurate, or cheaper, than the best single representation. Candidate directions are adaptive routing, evidence-aware context construction, and verified graph construction (proposal Section 5.5). Design earlier work so it serves RQ4: per-question results must be comparable across representations, and cost must be logged for every run.

### Planned architecture

Everything derives from one ground-truth OSM graph (nodes are intersections, edges are directed street segments with `name`, `length`, `oneway`, `highway`, and optionally `lanes`, `maxspeed`).

Five representations, all fed to the same LLM with the same answering prompt:

| ID | Representation | Derived from |
|---|---|---|
| M0 | Question only, no retrieval | nothing |
| M1 | Text RAG over generated street descriptions | text generated from the OSM graph |
| M2 | Graph RAG (LightRAG / HippoRAG 2) over an LLM-built graph | extracted from the M1 text; the true graph is deliberately withheld |
| M3 | k-hop subgraph of the OSM graph serialized into the prompt | OSM graph |
| M4 | Graph as a tool (`shortest_path`, `neighbors`, Cypher) | OSM graph |

Five question types, about 400 questions each: L1 attribute, L2 adjacency, L3 multi-step directional, L4 shortest legal route, L5 global aggregation.

Question generation pipeline: template → sample entities from the graph → compute the answer with code → LLM paraphrase → human check.

Ablations: A1 oracle evidence in the prompt (RQ2), A2 alignment of the M2 graph against the OSM graph (RQ3), A3 serialization format, A4 graph size.

### Design invariants

These come from the proposal and constrain any implementation:

- **Answers are always computed from the graph by code**, never written by a person or an LLM. The LLM is used only to paraphrase questions.
- **Each question records its required evidence** (the set of ground-truth nodes and edges needed to answer it). Evidence recall, context utilization, and the error funnel (entity linking → retrieval → usage, plus graph construction for M2) depend on this.
- **Development and test sets are split once and never mixed.** The RQ4 method is designed and tuned on the development set only; the test set is used once for the reported result.
- **Only the representation and retrieval vary.** The answering LLM, answering prompt, output format, and context budget stay constant across M0–M4.
- **No LLM-as-a-judge as a primary metric.** Scoring is exact match (L1, L3, L5), set F1 (L2), distance within a tolerance in metres plus path validity, legality, and optimality (L4).
- **Routes must respect edge direction.** One-way legality is checked explicitly, and wrong one-way direction is a tracked error type for M2.
- **Vietnamese street-name normalization** (diacritics, abbreviations such as "Đường 3/2" vs "Ba Tháng Hai", duplicate names across districts) is its own module, and entity-linking errors are reported separately.
- **Answer-generation code is verified by unit tests on small hand-checked subgraphs.**
- **LLMs are open-weight models served locally** (2–3 models, at least one around 7–8B), so cost metrics (prompt tokens, latency, index-building time and tokens) must be logged per run.
- Ground truth means "correct with respect to OSM", not the real world.

## Editing the proposal

`docs/thesis_proposal.md` follows conventions stated at its top: prior-work claims are cited as `[n]` against the References section, predictions are marked **Hypothesis**, and unfinalized design choices are marked **Planned**. Keep these when editing, and do not add claims about prior work without a reference.
