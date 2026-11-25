# EXP-001: Phase 1 pilot: four representations on 250 questions

| | |
|---|---|
| **Status** | pipeline trial on graph v1 (superseded by EXP-003, decision of 2026-11-25) |
| **Research question** | RQ1, with first evidence for RQ2 (error stages) and RQ3 (direction loss in the LLM-built graph) |
| **Approved by owner on** | 2026-11-24 (the owner delegated the full Phase 1 design and asked for it to be run without further approval steps; this protocol was fixed before any run) |

## 1. Question

On a small benchmark over the frozen District 1 graph, how do question answering accuracy and cost differ between no retrieval (M0), text RAG (M1), Graph RAG with an LLM-built graph (M2, LightRAG) and an OSM subgraph serialized into the prompt (M3), per question type?

## 2. Hypothesis

**Hypothesis** H1: M3 is more accurate than M1 and M2 on L2 (adjacency) and L3 (direction), because it shows the true edges with their direction.

**Hypothesis** H2: M2 is not more accurate than M1 on L3, and makes direction errors on L4, because LightRAG 1.5.8 stores an undirected graph (`networkx.Graph`) and asks the extractor for undirected relations, so direction survives only in description text.

**Hypothesis** H3: every representation is weak on L4 (shortest legal route) and L5 (aggregation), because these need computation over many edges that the answering LLM must do in its head.

**Hypothesis** H4: M0 is near the constant-guess baseline on every type.

A result rejects H1 if M3 accuracy on L2 or L3 is not higher than both M1 and M2 by more than the 95% confidence interval allows; likewise for the others. With 50 questions per type, only large differences (about 20 points or more) can be called; this is a pilot, so the main purpose is to see where errors concentrate.

## 3. Design

- **Independent variable:** representation, four levels: M0, M1, M2, M3.
- **Controlled variables:** answering LLM, answering prompt (`configs/prompts/answer_v1.yaml`, only the context differs), JSON output format per answer kind, context budget of 16,000 characters for M1, M2 and M3, temperature 0, seed 0.
- **Data:** benchmark `v0.2` (`data/benchmark/v0.2/`), 250 questions, 50 per type, all in the development split; graph `hcmc_district1_v2_p1`; text corpus `hcmc_district1_v2_p1_text_v1` (520 template-generated street descriptions, 631 chunks).
- **Question templates** (answers and evidence computed by code from the graph):
  - L1: one-way status of a street (25, balanced yes/no) and "is the street longer than N m" (25, balanced; N at least 25% away from the true length).
  - L2: the regular streets that meet a street (50; 2 to 8 streets per answer).
  - L3: at the intersection of street X and one-way street Y, the first regular street met when following Y in its legal direction (50).
  - L4: shortest legal car route between two intersections, as a street sequence and a length (50; 400 to 2,000 m, 2 to 5 streets).
  - L5: among the streets meeting S, how many are two-way along their whole length (25), and how many streets meet both S and T (25); answers stratified by value.
- **Models:**
  - Answering LLM: `qwen2.5:7b` (Q4_K_M) on Ollama 0.40.16, served as `qwen2.5-7b-ctx16k` (`num_ctx` 16384), OpenAI-compatible API.
  - LightRAG index LLM: `qwen2.5:14b` (Q4_K_M) as `qwen2.5-14b-ctx16k`; LightRAG query-time keyword extraction uses the answering LLM.
  - Embeddings for M1 and M2: `bge-m3` on Ollama.
- **Representation settings:**
  - M1: cosine top-30 chunks, added in rank order while they fit the budget.
  - M2: LightRAG 1.5.8, `mix` mode, `top_k` 40, `chunk_top_k` 20, entity and relation budgets 2,000 tokens each, total 8,000 tokens, no reranker, entity extraction without gleaning; the context string is cut at the budget.
  - M3: streets named in the question are found by exact name matching (`StreetLinker`); their 2-hop neighbourhood (ignoring direction) is serialized one segment per line with direction arrows, length and lanes, in breadth-first order until the budget.

## 4. Implementation

- Config: `configs/experiments/exp001.yaml`.
- Command: `uv run python scripts/run_experiment.py configs/experiments/exp001.yaml M0 M1 M2 M3`.
- Code version: the repository has no commit containing this code yet, so every run manifest records the commit marked `-dirty` and a SHA-256 over all files in `src/` (`source_tree_sha256`).

## 5. Evaluation

- **Metrics per question type and template** (no LLM judge):
  - L1, L3, L5: exact match (street names compared after normalization; a match without diacritics is accepted).
  - L2: set F1 (mean F1 is the score; accuracy counts F1 = 1).
  - L4: correct when a legal route exists that follows the predicted street sequence in order and is at most 2% longer than the true shortest route. Also reported: legality, wrong one-way direction, stated length within 10%, exact sequence.
- **Diagnostics:**
  - Evidence recall: share of required evidence edges present in the context. M1 uses the edges of the retrieved chunks; M2 counts an edge as present when a corpus sentence describing it appears verbatim in the context (knowledge graph entries are not counted); M3 uses the serialized edges.
  - Error stage of each wrong answer: output format, no context (M0), entity linking (M3 only), retrieval (evidence recall below 1), usage (all evidence present).
  - Cost per question: prompt and completion tokens, LLM calls, latency. M2 index cost: LLM calls, tokens, wall-clock time.
- **Uncertainty:** 95% Wilson intervals for accuracy per type; paired comparisons between representations on the same questions with an exact McNemar test. No correction for multiple comparisons; results are exploratory.
- **Failures:** an unparseable answer scores 0 and counts as an output-format error. A run that crashes is rerun as a new run id; LLM responses are cached, so finished questions are not recomputed.

## 6. Expected cost

Four runs of 250 questions, about 1,000 answering calls plus 250 keyword-extraction calls for M2. Measured throughput of the 7B model is about 450 prompt tokens/s, so roughly 3 to 4 hours in total. The LightRAG index over about 150,000 tokens of text with the 14B model is expected to take 2 to 4 hours.

## 7. Deviations

- 2026-11-24, before any M2 run: the M2 index is built from the corpus documents joined into 145 packs of at most 3,000 characters (`pack_documents`), not from 520 separate documents. Reason: LightRAG chunks each document on its own and every extraction call carries about 6,000 tokens of fixed prompt, so 520 short documents measured about 80 s per call (over 11 hours). The text is unchanged; only the chunk boundaries differ. Index id `hcmc_district1_v2_p1_text_v1_packed_qwen2.5-14b-ctx16k`.
- 2026-11-24, before any run: in benchmark v0.2, 56 of 250 paraphrases failed the automatic check (mostly L4 without "tuân thủ", and L3) and keep the template text. This is recorded in `data/benchmark/v0.2/manifest.json`.
- To keep latency and index-building time free of interference, runs are executed one at a time: M0, M1, M3 first, then the M2 index, then M2.
