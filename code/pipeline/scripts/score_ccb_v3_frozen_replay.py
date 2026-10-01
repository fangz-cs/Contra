#!/usr/bin/env python3
"""Replay frozen question trajectories against existing V3 match evidence.

This scorer performs no LLM calls. RITE uses the frozen candidate-to-Gold
votes. Baseline questions reuse their original V1 hit feedback for exact-shared
Gold and the frozen incremental votes for V3-added or revised Gold.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from forkcast.ccb_v3_harness import (
    HarnessError,
    index_tasks,
    load_plan,
    read_jsonl,
    sha256_bytes,
)
from forkcast.ccb_v3_scoring import per_task, summarize


QUESTION_PATTERN = re.compile(
    r"^\s*\[QUESTION\]\s*(.*?)\s*\[/QUESTION\]\s*$",
    re.DOTALL,
)
METHODS = ("RITE", "Vanilla LLM", "Ask-or-Assume", "ClarifyGPT")
DELTA_METRICS = (
    "KQR_micro",
    "TKQR",
    "ORA",
    "avg_question_turns",
    "zero_gold_ask_rate",
)


def index_unique(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        value = row.get(key)
        if not isinstance(value, str) or not value or value in output:
            raise HarnessError(f"{label}: invalid or duplicate {key}")
        output[value] = row
    return output


def atomic_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def extract_question(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = QUESTION_PATTERN.fullmatch(value)
    if match is None:
        return None
    question = match.group(1).strip()
    return question or None


def baseline_candidate_id(task_id: str, question: str) -> str:
    digest = hashlib.sha256(
        f"{task_id}\0{question}".encode("utf-8")
    ).hexdigest()[:24]
    return f"baseline::{task_id}::{digest}"


def load_official_metrics(release: Path):
    release = release.resolve()
    if not (release / "clarifycodebench/metrics.py").is_file():
        raise HarnessError(f"invalid ClarifyCodeBench release: {release}")
    sys.path.insert(0, str(release))
    try:
        return importlib.import_module("clarifycodebench.metrics")
    finally:
        sys.path.pop(0)


def load_gold(
    *,
    tasks: Mapping[str, Mapping[str, Any]],
    gold_path: Path,
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, str]],
    set[str],
]:
    rows = read_jsonl(gold_path)
    gold_order: dict[str, list[str]] = defaultdict(list)
    gold_by_question: dict[str, dict[str, str]] = defaultdict(dict)
    new_gold: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        gold_id = row.get("gold_id")
        item_id = row.get("item_id")
        question = row.get("question")
        if (
            not isinstance(gold_id, str)
            or gold_id in seen
            or not isinstance(item_id, str)
            or not item_id.startswith("v_task_")
            or not isinstance(question, str)
        ):
            raise HarnessError("invalid V3 Gold result")
        task_id = item_id[2:]
        if task_id not in tasks:
            raise HarnessError(f"{gold_id}: unknown task")
        if question in gold_by_question[task_id]:
            raise HarnessError(f"{task_id}: duplicate Gold question")
        seen.add(gold_id)
        gold_order[task_id].append(gold_id)
        gold_by_question[task_id][question] = gold_id
        if row.get("baseline_evaluation_source") == "new_v3_matcher":
            new_gold.add(gold_id)
    for task_id, task in tasks.items():
        gold_order.setdefault(task_id, [])
        gold_by_question.setdefault(task_id, {})
        expected = [
            str(pair["question"]) for pair in task.get("qa_pairs", [])
        ]
        observed = list(gold_by_question[task_id])
        if observed != expected:
            raise HarnessError(f"{task_id}: Gold order drift")
    if len(seen) != 463:
        raise HarnessError("V3 Gold census mismatch")
    return dict(gold_order), dict(gold_by_question), new_gold


def load_matched_pairs(
    path: Path,
    *,
    allowed_gold: set[str] | None = None,
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        candidate_id = row.get("candidate_id")
        gold_id = row.get("gold_id")
        votes = row.get("votes")
        matched = row.get("matched")
        key = (str(candidate_id), str(gold_id))
        if (
            not isinstance(candidate_id, str)
            or not isinstance(gold_id, str)
            or key in seen
            or not isinstance(votes, int)
            or isinstance(votes, bool)
            or not 0 <= votes <= 3
            or matched is not (votes >= 2)
            or (
                allowed_gold is not None
                and gold_id not in allowed_gold
            )
        ):
            raise HarnessError("invalid candidate-to-Gold pair vote")
        seen.add(key)
        if matched:
            output[candidate_id].add(gold_id)
    return dict(output)


def simulate_question(
    *,
    matched_gold: set[str],
    gold_order: Sequence[str],
    covered: set[str],
) -> int:
    for gold_id in gold_order:
        if gold_id not in matched_gold:
            continue
        if gold_id in covered:
            return 0
        covered.add(gold_id)
        return 1
    return 0


def rite_sequences(
    *,
    tasks: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
    matched: Mapping[str, set[str]],
    gold_order: Mapping[str, Sequence[str]],
    question_budget: int,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for task_id in tasks:
        covered: set[str] = set()
        hit_sequence: list[int] = []
        questions = plan["tasks"][task_id]["questions"][:question_budget]
        for question in questions:
            hit_sequence.append(
                simulate_question(
                    matched_gold=set(
                        matched.get(question["candidate_id"], set())
                    ),
                    gold_order=gold_order[task_id],
                    covered=covered,
                )
            )
        output[task_id] = {
            "question_count": len(questions),
            "hit_sequence": hit_sequence,
        }
    return output


def assistant_message_positions(
    record: Mapping[str, Any],
) -> list[int]:
    positions = [
        index
        for index, message in enumerate(record.get("messages") or [])
        if message.get("role") == "assistant"
    ]
    responses = record.get("response_contents") or []
    if len(positions) != len(responses):
        raise HarnessError(
            f"{record.get('task_id')}: assistant transcript drift"
        )
    return positions


def interactive_sequences(
    *,
    label: str,
    records: Mapping[str, Mapping[str, Any]],
    current_v1: Mapping[str, Mapping[str, Any]],
    legacy_v1: Mapping[str, Mapping[str, Any]],
    gold_order: Mapping[str, Sequence[str]],
    gold_by_question: Mapping[str, Mapping[str, str]],
    incremental_matches: Mapping[str, set[str]],
    question_budget: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    output: dict[str, dict[str, Any]] = {}
    diagnostics = {
        "original_shared_hits": 0,
        "original_removed_hits": 0,
        "incremental_v3_hits": 0,
    }
    for task_id, record in records.items():
        current_pairs = current_v1[task_id]["qa_pairs"]
        legacy_pairs = legacy_v1[task_id]["qa_pairs"]
        if len(current_pairs) != len(legacy_pairs):
            raise HarnessError(f"{task_id}: V1 Gold alignment drift")
        legacy_to_current = {
            str(legacy["question"]): str(current["question"])
            for current, legacy in zip(current_pairs, legacy_pairs)
        }
        hit_questions = {
            str(value) for value in record.get("hit_questions") or []
        }
        assigned_legacy: set[str] = set()
        positions = assistant_message_positions(record)
        responses = list(record.get("response_contents") or [])
        hit_dict = record.get("hit_dict") or {}
        covered: set[str] = set()
        hit_sequence: list[int] = []
        question_count = 0

        for turn, content in enumerate(responses):
            if question_count >= question_budget:
                break
            if str(turn) not in hit_dict:
                break
            question = extract_question(content)
            if question is None:
                question = str(content).strip()
            if not question:
                raise HarnessError(f"{task_id}: empty {label} question")
            question_count += 1
            candidate_id = baseline_candidate_id(task_id, question)
            matched_gold = set(incremental_matches.get(candidate_id, set()))

            if int(hit_dict[str(turn)]) == 1:
                message_index = positions[turn]
                messages = record["messages"]
                if (
                    message_index + 1 >= len(messages)
                    or messages[message_index + 1].get("role") != "user"
                ):
                    raise HarnessError(
                        f"{task_id}: missing judge feedback at turn {turn}"
                    )
                reply = str(messages[message_index + 1].get("content") or "")
                possible = [
                    str(pair["question"])
                    for pair in legacy_pairs
                    if pair["question"] in hit_questions
                    and reply.startswith(str(pair["answer"]))
                    and pair["question"] not in assigned_legacy
                ]
                if len(possible) != 1:
                    raise HarnessError(
                        f"{task_id}: ambiguous frozen hit at turn {turn}"
                    )
                legacy_question = possible[0]
                assigned_legacy.add(legacy_question)
                current_question = legacy_to_current[legacy_question]
                shared_gold = gold_by_question[task_id].get(
                    current_question
                )
                if shared_gold is None:
                    diagnostics["original_removed_hits"] += 1
                else:
                    diagnostics["original_shared_hits"] += 1
                    matched_gold.add(shared_gold)

            before = set(covered)
            hit = simulate_question(
                matched_gold=matched_gold,
                gold_order=gold_order[task_id],
                covered=covered,
            )
            if (
                hit
                and (covered - before)
                & incremental_matches.get(candidate_id, set())
            ):
                diagnostics["incremental_v3_hits"] += 1
            hit_sequence.append(hit)

        output[task_id] = {
            "question_count": question_count,
            "hit_sequence": hit_sequence,
        }
    return output, diagnostics


def clarifygpt_sequences(
    *,
    records: Mapping[str, Mapping[str, Any]],
    current_v1: Mapping[str, Mapping[str, Any]],
    gold_order: Mapping[str, Sequence[str]],
    gold_by_question: Mapping[str, Mapping[str, str]],
    incremental_matches: Mapping[str, set[str]],
    question_budget: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    output: dict[str, dict[str, Any]] = {}
    diagnostics = {
        "original_shared_match_occurrences": 0,
        "original_removed_match_occurrences": 0,
        "incremental_v3_hits": 0,
    }
    for task_id, record in records.items():
        questions = [
            str(value).strip()
            for value in record.get("questions") or []
            if str(value).strip()
        ]
        returned = list(record.get("qa_returned") or [])
        if len(returned) != len(questions):
            raise HarnessError(f"{task_id}: ClarifyGPT QA alignment drift")
        original_hits = {
            str(value)
            for value in record.get("matched_key_questions") or []
        }
        current_pairs = current_v1[task_id]["qa_pairs"]
        covered: set[str] = set()
        hit_sequence: list[int] = []

        for index, question in enumerate(questions[:question_budget]):
            candidate_id = baseline_candidate_id(task_id, question)
            matched_gold = set(incremental_matches.get(candidate_id, set()))
            answer = str(returned[index].get("a") or "")
            possible = [
                str(pair["question"])
                for pair in current_pairs
                if pair["question"] in original_hits
                and str(pair["answer"]) == answer
            ]
            if len(possible) > 1:
                raise HarnessError(
                    f"{task_id}: ambiguous ClarifyGPT match at {index}"
                )
            if possible:
                shared_gold = gold_by_question[task_id].get(possible[0])
                if shared_gold is None:
                    diagnostics[
                        "original_removed_match_occurrences"
                    ] += 1
                else:
                    diagnostics[
                        "original_shared_match_occurrences"
                    ] += 1
                    matched_gold.add(shared_gold)

            before = set(covered)
            hit = simulate_question(
                matched_gold=matched_gold,
                gold_order=gold_order[task_id],
                covered=covered,
            )
            if (
                hit
                and (covered - before)
                & incremental_matches.get(candidate_id, set())
            ):
                diagnostics["incremental_v3_hits"] += 1
            hit_sequence.append(hit)

        output[task_id] = {
            "question_count": min(len(questions), question_budget),
            "hit_sequence": hit_sequence,
        }
    return output, diagnostics


def score_sequences(
    *,
    method: str,
    sequences: Mapping[str, Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
    official_metrics: Any,
    question_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        sequence = sequences[task_id]
        hits = list(sequence["hit_sequence"])
        if len(hits) != sequence["question_count"]:
            raise HarnessError(f"{task_id}: hit-sequence length drift")
        record = {
            "task_id": task_id,
            "prompt": task["modified_content"],
            "hit_dict": {
                str(index): int(hit)
                for index, hit in enumerate(hits)
            },
            "turns": len(hits) + 1,
            "finished": True,
        }
        row = per_task(
            record=record,
            task=task,
            official_metrics=official_metrics,
            question_budget=question_budget,
        )
        rows.append({"method": method, **row, "hit_sequence": hits})
    summary = summarize(rows)
    summary["questions_asked"] = sum(
        int(row["question_turns"]) for row in rows
    )
    summary["tasks_asking"] = sum(
        int(row["question_turns"]) > 0 for row in rows
    )
    return rows, summary


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise HarnessError("invalid bootstrap percentile")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def cluster_bootstrap_deltas(
    *,
    rows: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 1:
        raise HarnessError("bootstrap sample count must be positive")
    aggregates: dict[str, dict[str, Counter[str]]] = {
        method: defaultdict(Counter) for method in METHODS
    }
    for row in rows:
        method = str(row["method"])
        cluster = str(row["source"])
        aggregate = aggregates[method][cluster]
        aggregate["tasks"] += 1
        aggregate["gold"] += int(row["K"])
        aggregate["hits"] += int(row["hits"])
        aggregate["tkqr"] += float(row["tkqr"])
        aggregate["ora"] += float(row["ora"])
        aggregate["questions"] += int(row["question_turns"])
        if int(row["K"]) == 0:
            aggregate["zero_tasks"] += 1
            aggregate["zero_asks"] += int(row["question_turns"]) > 0
    clusters = sorted(aggregates["RITE"])
    if len(clusters) != 80 or any(
        set(aggregates[method]) != set(clusters) for method in METHODS
    ):
        raise HarnessError("source-cluster coverage drift")

    def metrics(
        method: str, sampled_clusters: Sequence[str]
    ) -> dict[str, float]:
        total: Counter[str] = Counter()
        for cluster in sampled_clusters:
            total.update(aggregates[method][cluster])
        return {
            "KQR_micro": total["hits"] / total["gold"],
            "TKQR": total["tkqr"] / total["tasks"],
            "ORA": total["ora"] / total["tasks"],
            "avg_question_turns": (
                total["questions"] / total["tasks"]
            ),
            "zero_gold_ask_rate": (
                total["zero_asks"] / total["zero_tasks"]
            ),
        }

    rng = random.Random(seed)
    draws: dict[str, dict[str, list[float]]] = {
        baseline: {metric: [] for metric in DELTA_METRICS}
        for baseline in METHODS
        if baseline != "RITE"
    }
    for _ in range(samples):
        sampled = rng.choices(clusters, k=len(clusters))
        rite = metrics("RITE", sampled)
        for baseline in draws:
            other = metrics(baseline, sampled)
            for metric in DELTA_METRICS:
                draws[baseline][metric].append(
                    rite[metric] - other[metric]
                )

    output: dict[str, Any] = {}
    for baseline, metric_draws in draws.items():
        output[baseline] = {}
        for metric, values in metric_draws.items():
            point = (
                float(summaries["RITE"][metric])
                - float(summaries[baseline][metric])
            )
            output[baseline][metric] = {
                "RITE_minus_baseline": point,
                "ci95": [
                    percentile(values, 0.025),
                    percentile(values, 0.975),
                ],
            }
    return {
        "unit": "source_problem_cluster",
        "cluster_count": len(clusters),
        "samples": samples,
        "seed": seed,
        "comparisons": output,
    }


def render_summary(output: Mapping[str, Any]) -> str:
    lines = [
        "# ClarifyCodeBench V3 frozen-question replay",
        "",
        (
            "Question trajectories are capped at five and followed by one "
            "logical terminal response. No generation, verifier, judge, or "
            "code-generation calls were made."
        ),
        "",
        (
            "| Method | Gold hits | KQR | TKQR | ORA | Avg. questions "
            "| K=0 ask rate |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = output["methods"][method]
        lines.append(
            f"| {method} | {row['hits']}/{row['gold']} "
            f"| {100 * row['KQR_micro']:.2f}% "
            f"| {row['TKQR']:.4f} | {row['ORA']:.4f} "
            f"| {row['avg_question_turns']:.3f} "
            f"| {100 * row['zero_gold_ask_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## RITE paired cluster-bootstrap deltas",
            "",
        ]
    )
    for baseline, comparison in output[
        "paired_cluster_bootstrap"
    ]["comparisons"].items():
        kqr = comparison["KQR_micro"]
        tkqr = comparison["TKQR"]
        ora = comparison["ORA"]
        lines.append(
            f"- versus {baseline}: KQR "
            f"{100 * kqr['RITE_minus_baseline']:+.2f} pp "
            f"(95% CI [{100 * kqr['ci95'][0]:+.2f}, "
            f"{100 * kqr['ci95'][1]:+.2f}]); TKQR "
            f"{tkqr['RITE_minus_baseline']:+.4f} "
            f"[{tkqr['ci95'][0]:+.4f}, {tkqr['ci95'][1]:+.4f}]; "
            f"ORA {ora['RITE_minus_baseline']:+.4f} "
            f"[{ora['ci95'][0]:+.4f}, {ora['ci95'][1]:+.4f}]."
        )
    lines.extend(
        [
            "",
            (
                "This is an offline frozen-evidence replay, not a fresh run "
                "of the official GPT-4o question matcher."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--rite-gold-results", type=Path, required=True)
    parser.add_argument("--rite-pair-votes", type=Path, required=True)
    parser.add_argument("--v1-variants", type=Path, required=True)
    parser.add_argument("--legacy-v1-variants", type=Path, required=True)
    parser.add_argument("--vanilla-results", type=Path, required=True)
    parser.add_argument("--ask-or-assume-results", type=Path, required=True)
    parser.add_argument("--clarifygpt-results", type=Path, required=True)
    parser.add_argument("--baseline-gold-results", type=Path, required=True)
    parser.add_argument("--baseline-pair-votes", type=Path, required=True)
    parser.add_argument("--question-budget", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.question_budget != 5:
        raise HarnessError("the frozen comparison requires five questions")
    task_rows = read_jsonl(args.data)
    tasks = index_tasks(task_rows)
    if (
        len(tasks) != 419
        or sum(len(row.get("qa_pairs") or []) for row in task_rows) != 463
        or sum(not row.get("qa_pairs") for row in task_rows) != 91
    ):
        raise HarnessError("interactive V3 census mismatch")
    plan = load_plan(args.plan, dataset_path=args.data, tasks=tasks)
    official_metrics = load_official_metrics(args.release)

    gold_order, gold_by_question, new_gold = load_gold(
        tasks=tasks,
        gold_path=args.baseline_gold_results,
    )
    rite_gold_order, rite_gold_by_question, _ = load_gold(
        tasks=tasks,
        gold_path=args.rite_gold_results,
    )
    if (
        rite_gold_order != gold_order
        or rite_gold_by_question != gold_by_question
    ):
        raise HarnessError("RITE/baseline V3 Gold drift")

    rite_matches = load_matched_pairs(args.rite_pair_votes)
    incremental_matches = load_matched_pairs(
        args.baseline_pair_votes,
        allowed_gold=new_gold,
    )
    current_v1 = index_unique(
        read_jsonl(args.v1_variants), "task_id", "current V1 variants"
    )
    legacy_v1 = index_unique(
        read_jsonl(args.legacy_v1_variants),
        "task_id",
        "legacy V1 variants",
    )
    if set(current_v1) != set(tasks) or set(legacy_v1) != set(tasks):
        raise HarnessError("V1/V3 task coverage drift")
    vanilla = index_unique(
        read_jsonl(args.vanilla_results),
        "task_id",
        "Vanilla LLM records",
    )
    ask = index_unique(
        read_jsonl(args.ask_or_assume_results),
        "task_id",
        "Ask-or-Assume records",
    )
    clarify = index_unique(
        read_jsonl(args.clarifygpt_results),
        "task_id",
        "ClarifyGPT records",
    )
    if set(vanilla) != set(tasks) or set(ask) != set(tasks):
        raise HarnessError("interactive baseline coverage drift")
    if set(clarify) != set(tasks):
        raise HarnessError("ClarifyGPT coverage drift")

    sequences: dict[str, dict[str, dict[str, Any]]] = {
        "RITE": rite_sequences(
            tasks=tasks,
            plan=plan,
            matched=rite_matches,
            gold_order=gold_order,
            question_budget=args.question_budget,
        )
    }
    vanilla_sequences, vanilla_diagnostics = interactive_sequences(
        label="Vanilla LLM",
        records=vanilla,
        current_v1=current_v1,
        legacy_v1=legacy_v1,
        gold_order=gold_order,
        gold_by_question=gold_by_question,
        incremental_matches=incremental_matches,
        question_budget=args.question_budget,
    )
    ask_sequences, ask_diagnostics = interactive_sequences(
        label="Ask-or-Assume",
        records=ask,
        current_v1=current_v1,
        legacy_v1=legacy_v1,
        gold_order=gold_order,
        gold_by_question=gold_by_question,
        incremental_matches=incremental_matches,
        question_budget=args.question_budget,
    )
    clarify_sequences, clarify_diagnostics = clarifygpt_sequences(
        records=clarify,
        current_v1=current_v1,
        gold_order=gold_order,
        gold_by_question=gold_by_question,
        incremental_matches=incremental_matches,
        question_budget=args.question_budget,
    )
    sequences.update(
        {
            "Vanilla LLM": vanilla_sequences,
            "Ask-or-Assume": ask_sequences,
            "ClarifyGPT": clarify_sequences,
        }
    )

    all_rows: list[dict[str, Any]] = []
    method_summaries: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        rows, summary = score_sequences(
            method=method,
            sequences=sequences[method],
            tasks=tasks,
            official_metrics=official_metrics,
            question_budget=args.question_budget,
        )
        all_rows.extend(rows)
        method_summaries[method] = summary
    bootstrap = cluster_bootstrap_deltas(
        rows=all_rows,
        summaries=method_summaries,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    input_paths = {
        "data": args.data,
        "plan": args.plan,
        "rite_gold_results": args.rite_gold_results,
        "rite_pair_votes": args.rite_pair_votes,
        "v1_variants": args.v1_variants,
        "legacy_v1_variants": args.legacy_v1_variants,
        "vanilla_results": args.vanilla_results,
        "ask_or_assume_results": args.ask_or_assume_results,
        "clarifygpt_results": args.clarifygpt_results,
        "baseline_gold_results": args.baseline_gold_results,
        "baseline_pair_votes": args.baseline_pair_votes,
        "official_metrics": args.release / "clarifycodebench/metrics.py",
    }
    output = {
        "schema_version": "ccb-v3-frozen-question-replay-v1",
        "question_budget": args.question_budget,
        "logical_terminal_turn": True,
        "tasks": len(tasks),
        "gold": sum(len(task.get("qa_pairs") or []) for task in tasks.values()),
        "zero_gold_tasks": sum(
            not task.get("qa_pairs") for task in tasks.values()
        ),
        "methods": method_summaries,
        "paired_cluster_bootstrap": bootstrap,
        "baseline_reconstruction": {
            "Vanilla LLM": vanilla_diagnostics,
            "Ask-or-Assume": ask_diagnostics,
            "ClarifyGPT": clarify_diagnostics,
        },
        "evidence": {
            "RITE": (
                "frozen three-replicate V3 candidate-to-Gold pair votes"
            ),
            "baseline_shared_gold": (
                "original frozen V1 interactive/ClarifyGPT hit feedback"
            ),
            "baseline_new_or_revised_gold": (
                "frozen three-replicate incremental V3 pair votes"
            ),
            "fresh_model_calls": 0,
            "final_code_evaluated": False,
        },
        "hashes": {
            name: sha256_bytes(path.read_bytes())
            for name, path in input_paths.items()
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.out_dir / "comparison-task-scores.jsonl", all_rows)
    atomic_json(args.out_dir / "comparison-summary.json", output)
    (args.out_dir / "summary.md").write_text(
        render_summary(output),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
