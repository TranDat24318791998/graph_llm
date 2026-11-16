# Literature Survey: Candidate Method Directions for RQ4

**Thesis:** Which Graph Representation Helps LLMs Answer Best? (Trần Phan Quốc Đạt, Student ID 2470077)
**Date:** 16 November 2026
**Scope:** Work from 2023–2026 on arXiv, ACL Anthology (ACL, NAACL, EMNLP, TACL, workshops), NeurIPS, ICLR, ICML, VLDB.

> **Conventions.** "Peer-reviewed" means the paper appears in the proceedings of a reviewed venue. arXiv-only papers are marked as preprints. "Not found" always means *not found within the scope of this survey*, not that the work does not exist. Section 6 contains the author's own analysis and is labelled as such. Everything else is reported from the cited sources.

---

## 1. Context

The thesis compares five ways of giving a road-network graph (OpenStreetMap, Ho Chi Minh City) to an LLM to answer Vietnamese questions:

| ID | Representation |
|---|---|
| M0 | No retrieval |
| M1 | Text descriptions of streets with text RAG |
| M2 | Graph RAG over a graph built by an LLM from the text |
| M3 | Subgraph of the ground-truth graph placed in the prompt |
| M4 | LLM calls tools that query the graph |

RQ1–RQ3 measure and diagnose. **RQ4** asks for a new method that is more accurate or cheaper than the best single representation. Three candidate directions were surveyed:

| Direction | Idea |
|---|---|
| **A. Adaptive router** | Classify the question, pick the cheapest representation that is good enough, escalate to a stronger one when the answer is unreliable |
| **B. Evidence-aware context construction** | Prune and order the subgraph per question so the LLM can use information already in the prompt |
| **C. Verified graph construction** | Schema-guided extraction and checking of directed edges to fix errors in LLM-built graphs |

---

## 2. Summary

| Direction | Status in the literature | What remains open |
|---|---|---|
| **A** | Crowded in 2025–2026: routing between text RAG and Graph RAG, cost-aware RL routers (including one at EMNLP 2026), escalation with answer verification | No work found that routes among **text RAG, Graph RAG and graph-query tools** on the same knowledge source; all evaluations use Wikipedia, medical or legal QA, none with route or aggregation questions |
| **B** | Subgraph selection is well studied on Freebase/Wikidata; gold-subgraph datasets exist | Findings on order and format **conflict** across studies; the largest gold-subgraph dataset excludes aggregation questions; no work found that uses gold evidence to optimize context construction after retrieval |
| **C** | Schema-guided extraction, entity resolution, LLM triple filtering and task-driven RL exist | No work found that measures and **specifically corrects edge-direction errors** against a gold graph; authors in this area state that their gold graphs are incomplete or unreliable |

On the two papers recalled from memory: **Adaptive-RAG** (Jeong et al., NAACL 2024) and **Self-RAG** (Asai et al., ICLR 2024) are both confirmed.

---

## 3. Direction A — Adaptive router

### 3.1 Closest works

| Work | Venue | What it does | Evaluation data | Peer-reviewed |
|---|---|---|---|---|
| **Adaptive-RAG** (Jeong et al.) [A1] | NAACL 2024 | A smaller LM classifies question complexity and selects no retrieval, single-step retrieval, or multi-step retrieval. Labels are collected automatically from model outcomes | Open-domain QA datasets covering single- and multi-hop questions | Yes |
| **StructRAG** (Li et al.) [A3] | ICLR 2025 | A hybrid structure router (Qwen2-7B trained with DPO on 900 preference pairs) chooses how to structure documents per question: table, graph, algorithm, catalogue or chunk | Loong; Podcast Transcripts | Yes |
| **R²Adapter** [A4] | arXiv 2026 | A DeBERTa-v3 classifier routes each query to vanilla RAG or graph RAG before retrieval. Low-confidence graph-routed queries are rewritten by an LLM into relational triplets. Up to 59% reduction in graph usage | HotpotQA, 2WikiMultihopQA, MuSiQue (1,000 queries each) | No (preprint) |
| **GraphRAG-Router** (Fan et al.) [A5] | EMNLP 2026 | Hierarchical router trained with SFT and two-stage RL that selects both the GraphRAG method (GraphRAG, RAPTOR, HippoRAG2, HyperGraphRAG, LinearRAG) and the generator LLM size. A cost-aware reward reduces large-LLM use by about 30%. Vanilla RAG is only a baseline, not a routing option | NQ, PopQA, TriviaQA, HotpotQA, 2WikiMultiHopQA, MuSiQue | Yes |
| **RAGRouter-Bench** (Wang et al.) [A6] | arXiv 2026 | Benchmark of 7,727 queries for routing among five paradigms: LLM-only, NaiveRAG, GraphRAG, HybridRAG, IterativeRAG. Token cost is split into construction and per-query cost. No paradigm dominates; corpus structural metrics predict graph-retrieval effectiveness | MuSiQue, QuALITY, UltraDomain (legal), GraphRAG-Bench (medical) | No (preprint) |

**Related works**

| Work | Venue | Relevance |
|---|---|---|
| Self-RAG (Asai et al.) [A2] | ICLR 2024 | The model decides on demand when to retrieve and critiques its own output through reflection tokens |
| FLARE [A7] | EMNLP 2023 | Retrieval triggered during generation when the model is uncertain |
| DRAGIN [A8] | ACL 2024 | Dynamic retrieval based on the model's real-time information needs |
| SkewRoute (Wang et al.) [A9] | Findings of EMNLP 2025 | Training-free routing between small and large LLMs in KG-RAG using the skewness of retrieval scores. About 40% fewer large-LLM calls |
| RouteRAG (Guo et al.) [A10] | arXiv 2025 | RL-trained policy chooses text retrieval, graph retrieval or both, and when to stop |
| HybGRAG (Lee et al.) [A11] | ACL 2025 | LLM router between text and hybrid (graph + text) retrieval, with a critic that validates results and sends corrective feedback |
| A2RAG (Liu et al.) [A12] | arXiv 2026 | Escalating retrieval (local 1-hop → bridge discovery → global PPR) with answer verification ("Triple-Check"); 58% of HotpotQA queries resolved at the cheapest stage |
| BYOKG-RAG (Mavromatis et al.) [A13] | EMNLP 2025 | Several graph retrieval strategies including OpenCypher query execution, but all outputs are **merged**, not selected |

### 3.2 Answers to the survey questions

**Which signals are used for routing?**

- Question-complexity classifiers: Adaptive-RAG, R²Adapter, RAGRouter-Bench baselines (TF-IDF + SVM reached 0.928 macro-F1)
- Model self-assessment or uncertainty: Self-RAG, FLARE, DRAGIN
- Retrieval score distribution: SkewRoute
- LLM routers trained with RL or DPO: GraphRAG-Router, RouteRAG, StructRAG
- Offline corpus structure metrics (connectivity, relation diversity, density, clustering): proposed as router inputs in RAGRouter-Bench

**Has routing been done among text RAG, Graph RAG and tool calls on the same knowledge source?**

- Routing between text RAG and Graph RAG: **yes** (R²Adapter, RouteRAG, RAGRouter-Bench).
- Routing among text RAG, Graph RAG **and** graph-query tools on one source: **not found**. BYOKG-RAG executes Cypher queries but merges all strategies. StructRAG chooses a representation, but builds it from documents at inference time rather than calling tools on an existing graph.

**Joint optimization of accuracy and cost, or fallback when an answer is unreliable?**

- **Yes.** GraphRAG-Router uses a cost-aware reward; RAGRouter-Bench reports token savings; A2RAG escalates retrieval with answer verification; HybGRAG iterates with a critic.

### 3.3 Limitations stated by the authors

- **GraphRAG-Router:** scope is mainly Wikipedia-based QA; unclear whether it works on harder retrieval tasks; evaluated offline with fixed corpora and indices.
- **RAGRouter-Bench (baseline study):** classifiers use query text only, excluding corpus-side signals; synthetic queries may not reflect real user noise; out-of-distribution generalization untested.
- **R²Adapter:** performance bounded by the underlying systems; router labels tied to a specific retriever, requiring retraining if it changes; static rewriting threshold.
- **SkewRoute:** tested only on KG-RAG.
- **A2RAG:** benefits shrink as the graph becomes sparse or fragmented; experiments on subsets only.
- **HybGRAG:** retrieval modules are simple; limited domain adaptation.

### 3.4 What remains open for the thesis

1. Including **graph-query tools (M4)** in the routing space alongside text and Graph RAG.
2. Evaluating routing on **route-finding and aggregation questions**, which none of the works above use.
3. Deriving router training labels from **gold evidence** rather than only from run outcomes.

---

## 4. Direction B — Evidence-aware context construction

### 4.1 Closest works

| Work | Venue | What it does | Evaluation data | Peer-reviewed |
|---|---|---|---|---|
| **G-Retriever** (He et al.) [B1] | NeurIPS 2024 | Retrieves a subgraph from a textual graph by solving a Prize-Collecting Steiner Tree problem | Textual-graph QA including WebQSP | Yes |
| **SubgraphRAG** (Li et al.) [B2] | ICLR 2025 | Lightweight MLP with parallel triple scoring and directional structural distance encoding; subgraph size adjustable to the query and the LLM | WebQSP, CWQ (Freebase) | Yes |
| **Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?** (Ge et al.) [B3] | ACL 2025 | Compares random, BFS, DFS, PageRank and Personalized PageRank orders. BFS raised connectivity accuracy by about 11 points over random; global tasks (Hamiltonian path, topological sort, shortest path) are more order-sensitive than local tasks | Synthetic Erdős–Rényi graphs (5–15 nodes); citation networks | Yes |
| **Dissecting GraphRAG** (Nishida et al.) [B4] | TACL 2026 | Module-level analysis. Graph construction quality matters most; template-based context reports beat LLM-written summaries on recall and speed | CDR-QA, DocRED-QA (questions generated from gold graphs) | Yes |
| **Ground-Truth Subgraphs / GTSQA** (Cattaneo et al.) [B5] | arXiv 2025 (reviewed on OpenReview; venue not stated) | Generates questions with **ground-truth subgraphs** by executing SPARQL queries. Retrievers trained on ground-truth subgraphs beat those trained on shortest paths by 4–14% EM Hits | 32,099 questions from Wikidata (ogbl-wikikg2) | Unclear |

**Related works**

| Work | Venue | Relevance |
|---|---|---|
| Talk like a Graph (Fatemi et al.) [B6] | ICLR 2024 | Graph-to-text encoding changes accuracy by 4.8–61.8% depending on the task |
| Graph Linearization Methods (Xypolopoulos et al.) [B7] | arXiv | Degree-based edge ordering with node relabeling gave the largest gains; synthetic graphs only |
| The Answer Path and the Grounding Instruction (Canedo) [B8] | arXiv 2026 | Oracle subgraphs from gold SPARQL: the answer chain is what matters; distractor triples barely matter; serialization format, triple order and subgraph size show **no measurable effect** at multi-hop depth with current frontier LLMs |
| Is GraphRAG Needed? (Chen et al.) [B9] | GEM @ ACL 2026 | Retrieval–generation gap: context coverage 77.5% → 83.5% while correct selection stayed at 45–48%; relation-grouped format reduces tokens |
| Lost in the Middle (Liu et al.) [B10] | TACL 2024 | LLMs under-use information in the middle of long contexts |

### 4.2 Answers to the survey questions

**How do subgraph pruning and selection methods work, and on which graphs are they evaluated?**

Three families: combinatorial optimization (PCST in G-Retriever), per-triple scoring (SubgraphRAG), and agentic iterative exploration (e.g. BYOKG-RAG [A13]). Evaluations are dominated by **Freebase and Wikidata** (WebQSP, CWQ, GTSQA).

**How do order and format affect the LLM's use of information?** The findings **conflict**:

- On synthetic graph tasks with smaller or older models, order has a clear effect [B3, B7].
- On Wikidata KGQA with frontier LLMs, order, format and size show no measurable effect at multi-hop depth [B8].
- Relation-grouped formatting reduces tokens [B9]; template-based structured reports outperform LLM-written summaries [B4].
- No result was found for **directed spatial graphs with small open-weight models**, which is the thesis setting.

**Has gold evidence been used to measure or optimize context construction?**

- To **evaluate** retrieval and **train retrievers**: yes (GTSQA [B5]).
- As an **upper bound**: yes (Dissecting GraphRAG [B4] uses gold triples).
- As an **oracle** to study LLM use of context: yes (Answer Path [B8]).
- To **optimize context construction** (pruning, ordering) after retrieval: **not found**.

### 4.3 Limitations stated by the authors

- **GTSQA:** no cyclic subgraphs and no aggregation, filtering or negation questions; entity linking excluded; questions are LLM-generated and may carry model-specific style.
- **Ge et al.:** no rigorous theoretical explanation of order effects; limited range of graph structures.
- **Graph Linearization:** synthetic graphs only; mostly one model family.
- **Dissecting GraphRAG:** two domains only; template-generated questions; main experiments use gold triples, so the effect of real extraction is partly simulated.
- **Answer Path:** oracle subgraphs rather than real retrieval; small four-hop group; answer scorer not validated across response formats.

### 4.4 What remains open for the thesis

1. Testing order and format effects on a **directed spatial graph** with small open-weight models, where published findings disagree.
2. Context construction for **aggregation questions (L5)**, which GTSQA explicitly excludes.
3. Using gold evidence to **optimize** context construction, not only to evaluate it.

---

## 5. Direction C — Verified graph construction

### 5.1 Closest works

| Work | Venue | What it does | Evaluation | Peer-reviewed |
|---|---|---|---|---|
| **EDC: Extract, Define, Canonicalize** (Zhang & Soh) [C1] | EMNLP 2024 | Three phases: open information extraction, schema definition, post-hoc canonicalization. Works with a given schema or a self-generated one; a trained retriever adds relevant schema elements to the prompt | Three KG construction benchmarks | Yes |
| **KGGen** (Mo et al.) [C2] | NeurIPS 2025 | Extraction followed by iterative clustering to resolve duplicate entities and edges | MINE benchmark; LLM judge agreed with manual scoring 90.2% of the time | Yes |
| **AutoGraph-R1** (Tsang et al.) [C3] | ACL 2026 | Trains the graph constructor with RL, using the graph's usefulness in a RAG pipeline as the reward rather than intrinsic graph quality | Multiple QA benchmarks | Yes |
| **GraphJudge** (Huang et al.) [C4] | arXiv 2025 | Entity-centric text denoising, then a fine-tuned LLM judges and filters draft triples. Compared against gold KGs with soft-matching metrics | REBEL-Sub, GenWiki, SciERC, Re-DocRED subset | No (preprint) |
| **LLMs for KG Extraction: A Schema-Constrained Evaluation Framework** (Ilves et al.) [C5] | KG-LLM Workshop @ LREC 2026 | Exact-match triple precision/recall/F1 against gold, schema violations, missed and hallucinated triples. Triple F1 4.2–39.9% versus entity F1 74.1–84.0% | DocRED (10-document sample) | Yes (workshop) |

### 5.2 Mechanisms already in LightRAG and HippoRAG 2

| Framework | Construction mechanisms | Verification | Edge direction |
|---|---|---|---|
| **LightRAG** [C6] | LLM extraction of entities and relations with entity types; LLM profiling (key-value summaries); deduplication and merging of identical entities and relations; incremental update by union | No explicit verification step described | Not discussed |
| **HippoRAG 2** [C7] | Schema-free OpenIE triple extraction; phrase and passage nodes; synonym edges added when embedding similarity exceeds a threshold | LLM filtering of triples only **at query time** ("recognition memory"), not during indexing | Not discussed |

### 5.3 Answers to the survey questions

**Which techniques improve LLM-extracted graphs?**

- Schema definition and canonicalization (EDC)
- Entity and edge resolution by clustering (KGGen; deduplication in LightRAG; synonym edges in HippoRAG 2)
- LLM-based triple filtering (GraphJudge)
- Task-driven RL training of the constructor (AutoGraph-R1)
- Fine-tuned small extractors, which Dissecting GraphRAG [B4] found produce cleaner triples than LLM prompting, mainly through higher relation precision

**Has extraction error been measured against a gold graph and then corrected in a targeted way, especially for directed or asymmetric relations?**

- **Measurement** against gold: yes (GraphJudge, Ilves et al.).
- **Targeted correction of edge-direction errors: not found.** GraphJudge creates negative samples by replacing the tail entity, not by reversing direction. Ilves et al. separate predicate and argument errors but do not analyse direction as a category.

### 5.4 Limitations stated by the authors

- **GraphJudge:** gold KGs may be unreliable; a better benchmark for KG quality is needed.
- **Ilves et al.:** 61.5% of predicted triples in a manual review were absent from the gold standard, many of them supported by the text, so automatic triple F1 "should be interpreted as a lower bound"; only 10 documents.
- **KGGen:** entities and relations are still over- or under-deduplicated; general LLMs may lack domain knowledge.

### 5.5 What remains open for the thesis

1. In the thesis setup, the M1 text is **generated from the OSM graph**, so the gold graph is complete by construction. Every extracted triple can be judged correct or incorrect, which addresses the gold-graph limitation stated in [C4] and [C5].
2. **Edge-direction errors** (one-way streets) as an explicit error category with targeted correction.

---

## 6. Cross-cutting questions

**Has any work combined two or three of the directions?** Partial combinations exist:

| Combination | Works |
|---|---|
| Routing / escalation + answer verification | A2RAG [A12], HybGRAG [A11] |
| Routing + building a structure per question | StructRAG [A3] |
| Several retrieval strategies combined (merged, not routed) | BYOKG-RAG [A13] |

No work combining all three directions was found.

**Has any of this been applied to spatial graphs or road networks?**

- **Bilateral Spatial Reasoning about Street Networks** (Moratz et al., arXiv 2025) [D1]: Graph-RAG with qualitative spatial relations built from OSM for Hamburg and Münster. Success rose from 0% to 62.5% over 240 trials. The authors note the method is less successful on longer routes.
- **MapReason-OSM** (preprint 2026) [D2]: graph-verifiable decisions on OSM street maps.
- No work was found that applies routing, evidence-aware context construction or graph verification to road networks.

---

## 7. Implications for the thesis

> **Author's analysis.** The points below are the author's interpretation of the survey results, intended for discussion with the supervisor. They are not established findings.

### 7.1 Assessment of the three directions

| Direction | Novelty risk | Possible framing |
|---|---|---|
| **A** | High. Close to GraphRAG-Router (EMNLP 2026), R²Adapter, RouteRAG, A2RAG | Must be framed narrowly: tools (M4) in the routing space, route and aggregation questions, router labels from gold evidence |
| **B** | Medium. Novelty may be read as an ablation rather than a method | Resolve conflicting order/format findings on directed spatial graphs; context construction for aggregation questions |
| **C** | Lowest. Clearest gap, directly linked to RQ3 | Complete gold graph + edge-direction errors as a measured and corrected category |

### 7.2 What the benchmark should record for each question

Whichever direction is chosen, recording the following now avoids rebuilding the dataset later:

1. **Gold node and edge set, with direction** (needed for B and C).
2. **A flag for dependence on directed edges** (to measure direction errors in C).
3. **Question type and structural features**: number of hops, whether computation is required (routing signals for A).
4. **Outcome and cost of every representation M0–M4** for each question: correctness, tokens, latency (router labels for A).
5. **Entity-linking difficulty**: duplicate names, abbreviations (an error source shared by all directions).

---

## References

### Direction A
- [A1] S. Jeong et al. *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.* NAACL 2024. https://aclanthology.org/2024.naacl-long.389
- [A2] A. Asai et al. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html
- [A3] Z. Li et al. *StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization.* ICLR 2025. https://mlanthology.org/iclr/2025/li2025iclr-structrag
- [A4] *R²Adapter: A Routing and Rewriting Adapter for Efficient Hybrid RAG.* arXiv:2609.02894 (preprint). https://arxiv.org/html/2609.02894v1
- [A5] D. Fan, C. Ji, Z. Wang, T. Chen, Q. Tan. *GraphRAG-Router: Learning Cost-Efficient Routing over GraphRAGs and LLMs with Reinforcement Learning.* EMNLP 2026. https://aclanthology.org/2026.emnlp-main.2218/
- [A6] Z. Wang et al. *RAGRouter-Bench: A Dataset and Benchmark for Adaptive RAG Routing.* arXiv:2602.00296 (preprint). https://arxiv.org/html/2602.00296v1 — baseline study: *Lightweight Query Routing for Adaptive RAG: A Baseline Study on RAGRouter-Bench.* arXiv:2604.03455. https://arxiv.org/pdf/2604.03455
- [A7] Z. Jiang et al. *Active Retrieval Augmented Generation* (FLARE). EMNLP 2023. https://preview.aclanthology.org/setup/2023.emnlp-main.495
- [A8] W. Su et al. *DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models.* ACL 2024. https://preview.aclanthology.org/setup/2024.acl-long.702
- [A9] H. Wang et al. *SkewRoute: Training-Free LLM Routing for Knowledge Graph Retrieval-Augmented Generation via Score Skewness of Retrieved Context.* Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.606.pdf
- [A10] Y. Guo et al. *RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning.* arXiv:2512.09487 (preprint). https://arxiv.deeppaper.ai/papers/2512.09487v1
- [A11] M.-C. Lee et al. *HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases.* ACL 2025. https://aclanthology.org/2025.acl-long.43
- [A12] J. Liu et al. *A2RAG: Adaptive Agentic Graph Retrieval for Cost-Aware and Reliable Reasoning.* arXiv:2601.21162 (preprint). https://arxiv.org/html/2601.21162v2
- [A13] C. Mavromatis, S. Adeshina et al. *BYOKG-RAG: Multi-Strategy Graph Retrieval for Knowledge Graph Question Answering.* EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1417.pdf

### Direction B
- [B1] X. He et al. *G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering.* NeurIPS 2024.
- [B2] M. Li, S. Miao, P. Li. *Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation* (SubgraphRAG). ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/11e1900e680f5fe1893a8e27362dbe2c-Abstract-Conference.html
- [B3] Y. Ge et al. *Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?* ACL 2025. https://aclanthology.org/2025.acl-long.321.pdf
- [B4] N. Nishida et al. *Dissecting GraphRAG: A Modular Analysis of Knowledge Structuring for Factoid Question Answering.* TACL 2026. https://aclanthology.org/2026.tacl-1.29.pdf
- [B5] A. Cattaneo, C. Luschi, D. Justus. *Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs.* arXiv:2511.04473. https://arxiv.org/pdf/2511.04473
- [B6] B. Fatemi, J. Halcrow, B. Perozzi. *Talk like a Graph: Encoding Graphs for Large Language Models.* ICLR 2024.
- [B7] C. Xypolopoulos et al. *Graph Linearization Methods for Reasoning on Graphs with Large Language Models.* arXiv:2410.19494 (preprint). https://arxiv.org/pdf/2410.19494
- [B8] A. Canedo. *The Answer Path and the Grounding Instruction in LLM Question Answering over Knowledge Graphs.* arXiv:2609.10237 (preprint). https://arxiv.org/pdf/2609.10237
- [B9] L. Chen et al. *Is GraphRAG Needed? From Basic RAG to Graph-/Agentic Solutions with Context Optimization.* GEM Workshop @ ACL 2026. https://aclanthology.org/2026.gem-main.40/
- [B10] N. F. Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024.

### Direction C
- [C1] B. Zhang, H. Soh. *Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction.* EMNLP 2024. https://aclanthology.org/2024.emnlp-main.548/
- [C2] B. Mo et al. *KGGen: Extracting Knowledge Graphs from Plain Text with Language Models.* NeurIPS 2025. https://neurips.cc/virtual/2025/poster/117386
- [C3] H. T. Tsang et al. *AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction.* ACL 2026. https://preview.aclanthology.org/ingest-acl/2026.acl-long.1070/
- [C4] H. Huang et al. *Can LLMs be Good Graph Judge for Knowledge Graph Construction?* (GraphJudge). arXiv:2411.17388 (preprint). https://arxiv.org/html/2411.17388v4
- [C5] M. Ilves, E. Barbu, J. Übi. *Large Language Models for Knowledge Graph Extraction: A Schema-Constrained Evaluation Framework.* KG-LLM Workshop @ LREC 2026. https://aclanthology.org/2026.kallm-1.21.pdf
- [C6] Z. Guo et al. *LightRAG: Simple and Fast Retrieval-Augmented Generation.* Findings of EMNLP 2025. https://arxiv.org/abs/2410.05779
- [C7] B. J. Gutiérrez et al. *From RAG to Memory: Non-parametric Continual Learning for Large Language Models* (HippoRAG 2). arXiv:2502.14802. https://arxiv.org/html/2502.14802

### Spatial graphs
- [D1] R. Moratz et al. *Bilateral Spatial Reasoning about Street Networks: Graph-based RAG with Qualitative Spatial Representations.* arXiv:2512.15388 (preprint). https://arxiv.org/html/2512.15388
- [D2] S. Venkatanarayanan, C. P. Isaac. *MapReason-OSM: Graph-Verifiable Mobility Decisions from Street Maps.* Preprint, 2026.
