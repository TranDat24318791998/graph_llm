# Experiments

An experiment is one complete experimental design that produces a complete set of results intended for the thesis. Writing code, running tests and debugging on a handful of questions are not experiments.

## Rules

1. **Protocol first.** Every experiment has a folder `EXP-NNN-short-name/` with a `protocol.md` copied from `PROTOCOL_TEMPLATE.md`. No run starts until the project owner has approved the protocol and its status is `approved`.
2. **A protocol is not edited after approval.** If the design has to change, record the change and its reason under "Deviations", or open a new experiment.
3. **Runs are immutable.** Each run writes to `runs/<run_id>/` and is never modified. A rerun is a new run with a new id.
4. **Results are derived, not typed.** Tables and figures in `results/` are produced by scripts from `runs/`. Each one names the run ids it was built from.

## Layout

```
experiments/
  PROTOCOL_TEMPLATE.md
  EXP-001-short-name/
    protocol.md     # the approved design
    results/        # generated tables and figures
    notes.md        # observations and interpretation, written after the runs
runs/
  <run_id>/
    manifest.json       # RunManifest: config, git commit, data versions, seed
    predictions.jsonl   # one Prediction per question
    scores.jsonl        # one Score per question
    metrics.json        # aggregated metrics
```

A run id has the form `<experiment_id>_<representation>_<model>_<timestamp>`, for example `EXP-001_M1_qwen-7b_20261120T093000`.

## Index

| ID | Title | Research question | Status |
|---|---|---|---|
| EXP-001 | Phase 1 pilot: M0-M3 on 250 questions (graph v1) | RQ1 (RQ2, RQ3 first evidence) | pipeline trial (graph v1 superseded) |
| EXP-002 | Phase 1 follow-up: graph context formats (M3, M3N, M3T) | RQ1, RQ2, towards RQ4 | abandoned (superseded by EXP-003 before any run) |
| EXP-003 | Phase 1 main comparison on reference graph v2: M0, M1, M2, M3, M3N, M3T | RQ1 (RQ2, RQ3, RQ4 first evidence) | approved |
