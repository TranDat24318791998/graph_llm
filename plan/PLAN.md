# Thesis Plan

This file tracks the progress of the thesis project. The research design itself is in `docs/thesis_proposal.md`; this file only tracks what is done, what is next and what is still undecided.

**Last updated:** 2026-11-16

## How to maintain this file

- Update the status table and the task checkboxes whenever a task is finished or its scope changes.
- A milestone is `done` only when its exit criterion is met and the project owner has reviewed the deliverable.
- Record every decision that shapes the project in the decision log, with its date and reason.
- The thesis must contribute a solution, not only a comparison. RQ1-RQ3 are diagnostic; RQ4 (milestone P7) is the method contribution. Keep this in view when designing earlier milestones.
- P5, P6 and P7 contain experiments. Each needs a protocol in `experiments/` approved by the owner before any run (see `experiments/README.md`).

## Status overview

Status values: `not started`, `in progress`, `in review`, `done`.

| ID | Milestone | Deliverable | Needs GPU | Status |
|---|---|---|---|---|
| P0 | Project skeleton | Tooling, schemas, experiment workflow | No | in review |
| P1 | OSM road graph | Cleaned graph of one district, attribute-coverage report | No | in progress |
| P2 | Question set | Templates L1-L5, answer and evidence code, trial set of about 200 questions | Little | not started |
| P3 | Run and evaluation framework | Cached LLM client, runner, metrics per question type | Little | not started |
| P4 | Five representations | M0, M1, M3, M4, then M2 | Yes | not started |
| P5 | Pilot experiment | EXP-001 on about 200 questions, to validate the design | Yes | not started |
| P6 | Main experiment and ablations | 2,000-question benchmark, final models, A1-A4 | Yes | not started |
| P7 | Method design and evaluation (RQ4) | A method that beats the best single representation on accuracy or cost, evaluated on the held-out test set | Yes | not started |
| P8 | Analysis and writing | Thesis manuscript | No | not started |

**Current focus:** P1.

## Milestones

### P0: Project skeleton

Exit criterion: the owner has reviewed the skeleton, and the first commit of it is made.

- [x] `uv` project with ruff, mypy strict, pytest, pre-commit
- [x] Directory layout (`src/`, `configs/`, `scripts/`, `tests/`, `data/`, `experiments/`, `runs/`)
- [x] Draft schemas: `Question`, `Evidence`, `Answer`, `RunManifest`, `Prediction`, `Score`
- [x] Experiment workflow and protocol template
- [ ] Owner review
- [ ] Commit

### P1: OSM road graph

Exit criterion: a frozen, cleaned graph of the chosen area is on disk, its loading code is tested, and the coverage report is reviewed by the owner.

- [x] Confirm the study area and how its boundary is defined: fixed rectangle around District 1 (`configs/areas/hcmc_district1.yaml`)
- [x] Download the area with OSMnx and freeze the snapshot in `data/raw/hcmc_district1/` with its download date and checksum (2026-11-16: 2,024 nodes, 4,448 directed edges)
- [x] Attribute-coverage survey: `docs/reports/osm_coverage_hcmc_district1.md`
- [ ] Owner review of the coverage report
- [ ] Decide which attributes are usable for questions, based on the survey
- [ ] Decide how unnamed edges (14.5 percent of edges, 5 percent of length) and alleys (`Hẻm ...`, 215 of 534 names) are treated
- [ ] Street-name normalization: diacritics, abbreviations, duplicate names, unnamed segments
- [ ] Street model: rule for merging same-name segments into one street
- [ ] Schema for the processed graph
- [ ] Unit tests on small hand-checked subgraphs
- [ ] Coverage report in `docs/`

### P2: Question set

Exit criterion: a trial set of about 200 questions (about 40 per type) exists in `data/benchmark/`, with every answer and evidence set computed by tested code.

- [ ] Templates for L1-L5
- [ ] Entity sampling on the graph
- [ ] Answer computation per question type
- [ ] Evidence computation per question type
- [ ] Question features per question type (`QuestionFeatures`): hop count, computation required, dependence on edge direction, entity-linking challenges; each needs a precise definition per question type before it is computed
- [ ] Unit tests for answer and evidence code on hand-checked subgraphs
- [ ] LLM paraphrasing step
- [ ] Meaning-preservation check procedure
- [ ] Development and test split, stratified by question type and fixed once (the RQ4 method is tuned on development only)
- [ ] Trial benchmark v0.1

### P3: Run and evaluation framework

Exit criterion: one command takes a config and a benchmark version and produces a complete, valid run directory with scores.

- [ ] LLM client for an OpenAI-compatible endpoint, with a response cache
- [ ] Config schema and loading
- [ ] Runner that writes `manifest.json`, `predictions.jsonl`, `scores.jsonl`, `metrics.json`
- [ ] Answer parsing from raw LLM output
- [ ] Metrics: exact match (L1, L3, L5), set F1 (L2), distance tolerance and route validity, legality, optimality (L4)
- [ ] Evidence recall and context utilization
- [ ] Error-stage attribution
- [ ] Cost logging: prompt tokens, latency, index-building time and tokens

### P4: Five representations

Exit criterion: each representation answers the trial benchmark end to end through the P3 runner.

- [ ] M0: no retrieval
- [ ] M1: street descriptions generated from the graph, with text retrieval
- [ ] M3: k-hop subgraph serialized into the prompt
- [ ] M4: graph tools (`shortest_path`, `neighbors`, and others)
- [ ] M2: Graph RAG framework over the M1 text; measure indexing cost on a small subset first
- [ ] Shared answering prompt and output format across all five

### P5: Pilot experiment (EXP-001)

Exit criterion: results on the trial benchmark are reviewed with the owner, and the design is either confirmed or revised.

- [ ] Protocol drafted
- [ ] Protocol approved by the owner
- [ ] Runs completed
- [ ] Results and notes written
- [ ] Design review with the owner

### P6: Main experiment and ablations

Exit criterion: all results tables needed for RQ1-RQ3 are generated from immutable runs.

- [ ] Final model selection and dedicated serving (see open questions Q2 and Q3)
- [ ] Benchmark v1.0: about 2,000 generated questions and about 100 free-form questions
- [ ] Human quality control on a 10 percent sample, with Cohen's kappa
- [ ] Main experiment protocol approved, then run
- [ ] A1: oracle information
- [ ] A2: LLM-built graph against the OSM graph
- [ ] A3: serialization format
- [ ] A4: graph size

### P7: Method design and evaluation (RQ4)

Exit criterion: a method is evaluated once on the held-out test set against M0-M4, with a component ablation, and the result is reviewed with the owner.

Candidate directions (proposal Section 5.5): A adaptive routing, B evidence-aware context construction, C verified graph construction. They are not ranked. The direction is chosen after the pilot: C if M2 makes substantial edge-direction errors on naturally written text, otherwise a narrowed A; B is a component of either, not a direction on its own.

- [x] Literature survey covering all three candidate directions: `docs/rq4_method_directions_survey.md` (done by the owner, 2026-11-16; key citations still to be spot-checked against the sources)
- [ ] Confirm the direction after the pilot (P5), based on where errors concentrate
- [ ] Oracle-router upper bound computed from the main-experiment results
- [ ] Method design on the development set
- [ ] Method protocol approved by the owner
- [ ] Evaluation on the held-out test set: accuracy and cost against M0-M4
- [ ] Component ablation
- [ ] Generalization check on a second district

### P8: Analysis and writing

- [ ] Scripts that generate every thesis table and figure from `runs/`
- [ ] Error analysis
- [ ] Thesis manuscript

## Open questions

| ID | Question | Needed by | Status |
|---|---|---|---|
| Q1 | Study area: District 1 is proposed. Its boundary on OSM may have changed after the 2025 administrative merger; if so, define the area by ward boundaries or a fixed polygon. | P1 | resolved 2026-11-16: a fixed rectangle around District 1; parts of neighbouring districts are acceptable |
| Q2 | Which 2-3 open-weight models are used for the thesis experiments? They need good Vietnamese and tool calling. | P6 | open |
| Q3 | Dedicated model serving for the main experiment: the shared vLLM server serves other workloads and is not under our control, and the disk has about 89 GB free. | P6 | open |
| Q4 | Which Graph RAG framework is used for M2: LightRAG, HippoRAG 2, or both? | P4 | open |
| Q5 | When is the novelty survey for all three RQ4 candidates (A, B, C) done? Doing it early is recommended, because the result can change the benchmark design. | P2 | resolved 2026-11-16: see `docs/rq4_method_directions_survey.md` |
| Q6 | Which RQ4 direction is pursued? Not ranked in advance; decided by the conditions in proposal Section 5.5 once the pilot results exist. | P7, after P5 | open |
| Q7 | Is the M1 text written by templates or rewritten by an LLM into natural prose? This decides whether direction C is defensible, since extraction from regular template text may be nearly error-free. To be discussed with the supervisor. | P4 | open |

## Risks

| Risk | Mitigation |
|---|---|
| Heavy load on the shared vLLM server slows down other users | Use it for development and the pilot only; cache all LLM calls; set up dedicated serving before P6 |
| Disk space limits the choice of models | Resolve Q3 before P6; download one model at a time |
| M2 index building is expensive | Implement M2 last; measure cost on a small subset before the full build |
| OSM attributes are incomplete for the study area | Coverage survey in P1 decides which attributes are used |
| M4 dominates on accuracy, leaving the RQ4 method no room to improve accuracy | Target cost and latency with direction A, or the documents-only setting with direction C |
| The RQ4 method overfits the benchmark | Development and test split fixed in P2; second district for generalization |
| The chosen RQ4 direction is not novel | Literature survey before the design is fixed |

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-11-16 | Code, comments and repository documents are in English; no emoji | Owner requirement |
| 2026-11-16 | Stack: Python 3.12, `uv`, ruff, mypy strict, pytest, pydantic v2 schemas | Explicit, typed, reproducible code |
| 2026-11-16 | Experiments are tracked with files (`experiments/` protocols, immutable `runs/`), without MLflow or DVC for now | Fewer dependencies; sufficient at the planned scale |
| 2026-11-16 | Every experiment needs an owner-approved protocol before any run | Clean and defensible experiments |
| 2026-11-16 | Development and the pilot use the shared vLLM server (Qwen3-32B) as a client only | Final models are not chosen yet; nothing outside the project may be modified |
| 2026-11-16 | Milestone order as listed above | Agreed with the owner |
| 2026-11-16 | Study area is a fixed rectangle around District 1 (west 106.680, south 10.755, east 106.713, north 10.797), drivable network, OSMnx simplification on | The owner does not need the exact administrative boundary; a fixed rectangle is reproducible and independent of how OSM draws district borders |
| 2026-11-16 | RQ4 direction is not ranked in advance; it is chosen by stated conditions after the pilot | The survey shows where gaps exist, but which direction is worth pursuing depends on pilot data |
| 2026-11-16 | Every question records the five items of survey Section 7.2 (directed evidence, direction dependence, structural features, per-representation outcome and cost, entity-linking difficulty) | Keeps all three RQ4 directions open without regenerating the benchmark |
| 2026-11-16 | RQ4 added: design a method from the RQ1-RQ3 findings that is more accurate or cheaper than the best single representation | The thesis must contribute a solution with novelty, not only compare existing methods |
