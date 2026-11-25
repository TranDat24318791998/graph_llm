"""Run one representation over a benchmark and write an immutable run directory.

A run directory holds manifest.json, predictions.jsonl, scores.jsonl and
metrics.json (see experiments/README.md). Every representation goes through the
same steps: build a context, ask the answering LLM with the shared prompt, parse,
score, and attribute errors to a pipeline stage.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from graph_llm.evaluation.answering import build_messages, parse_answer
from graph_llm.evaluation.scoring import score_answer
from graph_llm.llm.client import LLMClient
from graph_llm.questions.graph_view import GraphView, load_graph_view
from graph_llm.representations.corpus_store import load_corpus
from graph_llm.representations.graph_linked_text import graph_linked_chunks
from graph_llm.representations.lightrag_m2 import LightRAGQuerySettings, LightRAGRetriever
from graph_llm.representations.linker import StreetLinker
from graph_llm.representations.subgraph import NodeNaming, build_subgraph_context
from graph_llm.representations.text_corpus import segment_sentences, street_header
from graph_llm.representations.text_rag import retrieve_chunks
from graph_llm.schemas.experiment import ExperimentConfig, GroupMetrics, RunMetrics
from graph_llm.schemas.prompt import AnswerPromptConfig
from graph_llm.schemas.question import EdgeRef, Question
from graph_llm.schemas.run import ErrorStage, Prediction, Representation, RunManifest, Score

PROCESSED_DATA_DIRECTORY = Path("data/processed")
RAW_DATA_DIRECTORY = Path("data/raw")
BENCHMARK_DIRECTORY = Path("data/benchmark")
LIGHTRAG_DIRECTORY = Path("data/processed/lightrag")
RUNS_DIRECTORY = Path("runs")
NON_STREET_SLOTS = {"start_node_ids", "end_node_ids", "threshold_m"}


@dataclass(frozen=True)
class RetrievedContext:
    """The context of one question and what it cost to build."""

    text: str
    covered_edges: set[EdgeRef] | None
    linked_street_ids: set[str] | None
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


Retriever = Callable[[Question], RetrievedContext]


def load_questions(benchmark_version: str) -> list[Question]:
    """Load the questions of a benchmark version."""
    path = BENCHMARK_DIRECTORY / benchmark_version / "questions.jsonl"
    return [
        Question.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]


def source_tree_sha256() -> str:
    """Return one checksum over all library source files, to identify the code of a run."""
    digest = hashlib.sha256()
    for path in sorted(Path("src").rglob("*.py")):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit() -> str:
    """Return the current commit, marked dirty if the working tree has changes."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    return f"{commit}-dirty" if status.strip() else commit


def corpus_statements(view: GraphView) -> list[tuple[str, list[EdgeRef]]]:
    """Return every sentence group of the corpus with the edges it describes.

    Used to measure evidence recall of M2, whose context quotes source text that
    LightRAG has re-chunked: an edge counts as present when a statement that
    describes it appears verbatim in the context.
    """
    statements: list[tuple[str, list[EdgeRef]]] = []
    for street_id in sorted(view.streets):
        statements.append((street_header(view, street_id), list(view.streets[street_id].edges)))
        statements += segment_sentences(view, street_id)
    return statements


def make_retriever(
    representation: Representation,
    config: ExperimentConfig,
    view: GraphView,
    linker: StreetLinker,
    client: LLMClient,
) -> tuple[Retriever, Callable[[], None]]:
    """Return the context builder of a representation and a function that releases it."""
    budget = config.context_budget_characters
    if representation is Representation.M0_NO_RETRIEVAL:
        return (lambda question: RetrievedContext("", None, None)), lambda: None

    if representation is Representation.M1_TEXT_RAG:
        corpus = load_corpus(config.corpus_id)

        def m1(question: Question) -> RetrievedContext:
            text, covered = retrieve_chunks(client, corpus, question.text, config.m1_top_k, budget)
            return RetrievedContext(text, covered, None)

        return m1, lambda: None

    if representation is Representation.M3_SUBGRAPH:

        def m3(question: Question) -> RetrievedContext:
            linked = linker.link(question.text)
            text, covered = build_subgraph_context(view, linked, config.m3_hops, budget)
            return RetrievedContext(text, covered, linked)

        return m3, lambda: None

    if representation is Representation.M3_SUBGRAPH_NAMED_NODES:

        def m3n(question: Question) -> RetrievedContext:
            linked = linker.link(question.text)
            text, covered = build_subgraph_context(
                view, linked, config.m3_hops, budget, NodeNaming.INTERSECTION_NAMES
            )
            return RetrievedContext(text, covered, linked)

        return m3n, lambda: None

    if representation is Representation.M3_GRAPH_LINKED_TEXT:
        linked_corpus = load_corpus(config.corpus_id)

        def m3t(question: Question) -> RetrievedContext:
            linked = linker.link(question.text)
            text, covered = graph_linked_chunks(
                view, linked_corpus, linked, config.m3t_hops, budget
            )
            return RetrievedContext(text, covered, linked)

        return m3t, lambda: None

    if representation is Representation.M2_LLM_BUILT_GRAPH:
        settings = config.lightrag
        retriever = LightRAGRetriever(
            LIGHTRAG_DIRECTORY / config.lightrag_index_id,
            config.answer_llm,
            config.embedding_model,
            LightRAGQuerySettings(**settings.model_dump()),
        )
        statements = corpus_statements(view)

        def m2(question: Question) -> RetrievedContext:
            before = retriever.cost_log.snapshot()
            text = retriever.context(question.text)[:budget]
            after = retriever.cost_log.snapshot()
            covered = {edge for sentence, edges in statements if sentence in text for edge in edges}
            return RetrievedContext(
                text,
                covered,
                None,
                llm_calls=after[0] - before[0],
                prompt_tokens=after[1] - before[1],
                completion_tokens=after[2] - before[2],
            )

        return m2, retriever.close

    raise ValueError(f"Representation {representation} is not part of this experiment.")


def evidence_recall(question: Question, covered: set[EdgeRef] | None) -> float | None:
    """Return the share of required evidence edges present in the context."""
    if covered is None or not question.evidence.edges:
        return None
    present = sum(1 for edge in question.evidence.edges if edge in covered)
    return present / len(question.evidence.edges)


def attribute_error(
    representation: Representation,
    question: Question,
    parsed: bool,
    correct: bool,
    recall: float | None,
    linked: set[str] | None,
    linker: StreetLinker,
) -> ErrorStage:
    """Assign a wrong answer to the first pipeline stage that failed."""
    if correct:
        return ErrorStage.NONE
    if not parsed:
        return ErrorStage.OUTPUT_FORMAT
    if representation is Representation.M0_NO_RETRIEVAL:
        return ErrorStage.NO_CONTEXT
    if linked is not None:
        for slot, value in question.slots.items():
            if slot not in NON_STREET_SLOTS and not linker.resolve(value) & linked:
                return ErrorStage.ENTITY_LINKING
    if recall is not None and recall < 1.0:
        return ErrorStage.RETRIEVAL
    return ErrorStage.USAGE


def group_metrics(pairs: list[tuple[Prediction, Score]]) -> GroupMetrics:
    """Aggregate predictions and scores of one group."""
    count = len(pairs)
    recalls = [s.evidence_recall for _, s in pairs if s.evidence_recall is not None]
    sub_scores: dict[str, list[float]] = defaultdict(list)
    for _, score in pairs:
        for name, value in score.sub_scores.items():
            sub_scores[name].append(value)
    return GroupMetrics(
        question_count=count,
        accuracy=sum(s.is_correct for _, s in pairs) / count,
        mean_score=sum(s.score for _, s in pairs) / count,
        mean_evidence_recall=sum(recalls) / len(recalls) if recalls else None,
        parse_failure_rate=sum(p.parsed_answer is None for p, _ in pairs) / count,
        mean_prompt_tokens=sum(p.prompt_tokens for p, _ in pairs) / count,
        mean_completion_tokens=sum(p.completion_tokens for p, _ in pairs) / count,
        mean_llm_calls=sum(p.llm_call_count for p, _ in pairs) / count,
        mean_latency_s=sum(p.latency_s for p, _ in pairs) / count,
        error_stage_counts=dict(Counter(s.error_stage.value for _, s in pairs)),
        sub_score_means={name: sum(v) / len(v) for name, v in sorted(sub_scores.items())},
    )


def compute_metrics(
    run_id: str, questions: list[Question], predictions: list[Prediction], scores: list[Score]
) -> RunMetrics:
    """Aggregate a run overall, per question type and per template."""
    rows = list(zip(questions, predictions, scores, strict=True))
    by_type: dict[str, list[tuple[Prediction, Score]]] = defaultdict(list)
    by_template: dict[str, list[tuple[Prediction, Score]]] = defaultdict(list)
    for question, prediction, score in rows:
        by_type[question.question_type.value].append((prediction, score))
        by_template[question.template_id].append((prediction, score))
    return RunMetrics(
        run_id=run_id,
        overall=group_metrics([(p, s) for _, p, s in rows]),
        by_question_type={key: group_metrics(value) for key, value in sorted(by_type.items())},
        by_template={key: group_metrics(value) for key, value in sorted(by_template.items())},
    )


def run_representation(
    config: ExperimentConfig, representation: Representation, progress: bool = True
) -> RunMetrics:
    """Run one representation over the whole benchmark and write its run directory."""
    prompt = AnswerPromptConfig.model_validate(
        yaml.safe_load(Path(config.prompt_config_path).read_text(encoding="utf-8"))
    )
    view = load_graph_view(PROCESSED_DATA_DIRECTORY, RAW_DATA_DIRECTORY, config.graph_version)
    linker = StreetLinker.from_view(view)
    categories = config.question_linker_categories
    question_linker = StreetLinker.from_view(
        view, frozenset(categories) if categories is not None else None
    )
    client = LLMClient(config.answer_llm)
    questions = load_questions(config.benchmark_version)
    started_at = datetime.now(UTC)
    model_slug = config.answer_llm.model.replace(":", "-").replace("/", "-")
    run_id = (
        f"{config.experiment_id}_{representation.value}_{model_slug}_{started_at:%Y%m%dT%H%M%S}"
    )
    run_directory = RUNS_DIRECTORY / run_id
    run_directory.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_id,
        experiment_id=config.experiment_id,
        representation=representation,
        model_name=config.answer_llm.model,
        benchmark_version=config.benchmark_version,
        graph_version=config.graph_version,
        git_commit=git_commit(),
        seed=config.answer_llm.seed,
        config={
            **config.model_dump(mode="json"),
            "prompt": prompt.model_dump(),
            "source_tree_sha256": source_tree_sha256(),
        },
        created_at=started_at,
    )
    (run_directory / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )

    retrieve, release = make_retriever(representation, config, view, question_linker, client)
    predictions: list[Prediction] = []
    scores: list[Score] = []
    predictions_file = (run_directory / "predictions.jsonl").open("w", encoding="utf-8")
    scores_file = (run_directory / "scores.jsonl").open("w", encoding="utf-8")
    try:
        for index, question in enumerate(questions, start=1):
            question_started = time.perf_counter()
            context = retrieve(question)
            retrieval_seconds = time.perf_counter() - question_started
            result = client.chat(build_messages(prompt, question, context.text))
            parsed = parse_answer(question.answer.kind, result.text)
            correct, score_value, sub_scores = score_answer(view, linker, question, parsed)
            recall = evidence_recall(question, context.covered_edges)
            prediction = Prediction(
                run_id=run_id,
                question_id=question.question_id,
                context=context.text,
                raw_response=result.text,
                parsed_answer=parsed,
                prompt_tokens=result.prompt_tokens + context.prompt_tokens,
                completion_tokens=result.completion_tokens + context.completion_tokens,
                llm_call_count=1 + context.llm_calls,
                latency_s=retrieval_seconds + result.latency_s,
            )
            score = Score(
                run_id=run_id,
                question_id=question.question_id,
                is_correct=correct,
                score=score_value,
                evidence_recall=recall,
                error_stage=attribute_error(
                    representation,
                    question,
                    parsed is not None,
                    correct,
                    recall,
                    context.linked_street_ids,
                    linker,
                ),
                sub_scores=sub_scores,
            )
            predictions.append(prediction)
            scores.append(score)
            predictions_file.write(prediction.model_dump_json() + "\n")
            scores_file.write(score.model_dump_json() + "\n")
            predictions_file.flush()
            scores_file.flush()
            if progress and index % 25 == 0:
                done = sum(s.is_correct for s in scores)
                print(f"{run_id}: {index}/{len(questions)} done, {done} correct", flush=True)
    finally:
        predictions_file.close()
        scores_file.close()
        release()

    metrics = compute_metrics(run_id, questions, predictions, scores)
    (run_directory / "metrics.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    return metrics
