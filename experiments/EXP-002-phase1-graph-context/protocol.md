# EXP-002: Phase 1 follow-up: how graph context is presented

| | |
|---|---|
| **Status** | abandoned before any run: superseded by EXP-003 on graph v2 (2026-11-25) |
| **Research question** | RQ1 and RQ2 (usage errors), first step towards RQ4 (Phase 1 item 3, "one easy-to-implement method") |
| **Approved by owner on** | 2026-11-24 (the owner delegated the full Phase 1 design; this protocol was written after the EXP-001 runs of M0, M1 and M3 and before any EXP-002 run) |

## 1. Question

In EXP-001, M3 (the true OSM subgraph) put almost all required evidence into the context (evidence recall 0.98 on L2, 1.0 on L3) but was much less accurate than M1 (text RAG): L2 accuracy 0.04 against 0.62, L3 0.00 against 0.12. In 77 of 250 answers the model returned local node ids such as "n7" instead of street names. EXP-001 also showed that the question linker matched the fixed sentence "Không tính hẻm và đường nội bộ" to an internal road named "Đường Nội Bộ" in 150 of 250 questions, which added an unrelated neighbourhood to every such M3 context.

This experiment asks whether these are failures of presentation rather than of content. It compares three ways of putting graph-selected evidence into the same prompt.

## 2. Hypothesis

**Hypothesis** H5: naming nodes by the streets that meet there (M3N) is more accurate than local node ids (M3) on L2 and L3, and removes node-id answers. Rejected if M3N is not more accurate than M3 on L2 or L3 (exact McNemar test, p < 0.05) or if node-id answers remain above 5% of answers.

**Hypothesis** H6: selecting the M1 text through the street graph (M3T) is at least as accurate as M1 on every type, and more accurate on L5, because the graph puts the documents of all neighbouring streets into the context. Rejected if M3T is less accurate than M1 on any type, or not more accurate on L5.

**Hypothesis** H7: restricting the question linker to regular streets changes M3 accuracy by less than the 95% interval, because the spurious seed only added extra context. This compares the EXP-002 M3 run with the EXP-001 M3 run.

This is exploratory and uses the same 250 development questions as EXP-001; it is designed after seeing EXP-001 results, so any effect found here is a lead for RQ4, not a confirmed result. The test split is not used.

## 3. Design

- **Independent variable:** representation, three levels:
  - **M3** (as in EXP-001): 2-hop subgraph, nodes as local ids `n1, n2, ...`.
  - **M3N**: the same edges in the same order, but every node is written as the names of the streets that meet there in square brackets (`[Lê Lợi / Pasteur]`); a node where no named street meets is `[điểm k]`; two nodes with the same names get a suffix `#2`. The header line explains the notation. Because names are longer, fewer edges fit into the budget on large neighbourhoods (measured before the runs: evidence recall L4 0.70 against 0.83 for M3).
  - **M3T** (graph-linked text): the template-generated street documents of the M1 corpus, chosen by the graph: first the documents of the streets linked in the question, then those of the streets that meet them, then those one street further (2 street-level hops). Within a hop regular streets come first, then by name. Whole chunks are added while they fit the budget. M1 and M3T read the same text; only the selection differs (embedding similarity against the graph).
- **Change shared by all three:** the question linker links only regular streets (`question_linker_categories: [regular]`). Phase 1 questions name only regular streets, so no slot is lost; scoring still resolves predicted names against all streets.
- **Controlled variables:** as in EXP-001: answering LLM, answering prompt `answer_v1`, JSON output format, context budget 16,000 characters, temperature 0, seed 0.
- **Data:** benchmark `v0.2`, 250 development questions; graph `hcmc_district1_v2_p1`; corpus `hcmc_district1_v2_p1_text_v1`.
- **Models:** answering LLM `qwen2.5-7b-ctx16k` on Ollama, as in EXP-001.

## 4. Implementation

- Config: `configs/experiments/exp002.yaml`.
- Code: `NodeNaming.INTERSECTION_NAMES` in `src/graph_llm/representations/subgraph.py`; `src/graph_llm/representations/graph_linked_text.py`; `StreetLinker.from_view(view, categories)`.
- Command: `uv run python scripts/run_experiment.py configs/experiments/exp002.yaml M3 M3N M3T`, run after the EXP-001 M2 run, one at a time.
- Code version: recorded in every run manifest (commit marked `-dirty` and `source_tree_sha256`).

## 5. Evaluation

- Metrics, diagnostics and error stages exactly as in EXP-001 Section 5.
- Additional count: answers that contain a local node id (`"n<digits>"`).
- Comparisons: M3N against M3 and M3T against M1 (EXP-001 run), paired exact McNemar tests per type; EXP-002 M3 against EXP-001 M3 for H7.
- Known limitation of evidence recall for text contexts (M1, M3T): a chunk covers only the edges of its own street, so a street named in another street's sentence (for example the answer street of an L3 question) is not counted as present even when the text states the fact.

## 6. Expected cost

Three runs of 250 questions with the 7B model, about 1.5 to 2.5 hours in total. No index building.

## 7. Deviations

None so far.
