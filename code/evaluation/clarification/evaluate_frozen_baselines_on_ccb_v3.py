#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import audit_agent2_resolution_spans as util
import evaluate_frozen_av_on_ccb_v3 as av


SCHEMA_MANIFEST = "ccb-v3-frozen-baseline-evaluation-manifest-v1"
SCHEMA_MATCH = "ccb-v3-frozen-baseline-pair-vote-v1"
METHODS = ("Vanilla LLM", "Ask-or-Assume", "ClarifyGPT")
QUESTION_PATTERN = re.compile(
    r"^\s*\[QUESTION\]\s*(.*?)\s*\[/QUESTION\]\s*$",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-actionable", type=Path, required=True)
    parser.add_argument("--v1-variants", type=Path, required=True)
    parser.add_argument("--legacy-v1-variants", type=Path, required=True)
    parser.add_argument("--v1-v3-mapping", type=Path, required=True)
    parser.add_argument("--vanilla-results", type=Path, required=True)
    parser.add_argument("--ask-or-assume-results", type=Path, required=True)
    parser.add_argument("--clarifygpt-results", type=Path, required=True)
    parser.add_argument("--av-v3-summary", type=Path, required=True)
    parser.add_argument("--av-v3-gold-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=24)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.replicates != 3:
        raise av.EvaluationError("the frozen protocol requires 3 replicates")
    if args.candidate_batch_size < 1:
        raise av.EvaluationError("candidate batch size must be positive")
    return args


def extract_question(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    match = QUESTION_PATTERN.fullmatch(text)
    if not match:
        return None
    question = match.group(1).strip()
    return question or None


def trajectory_questions(row: Mapping[str, Any]) -> list[str]:
    output: list[str] = []
    for response in row.get("response_contents") or []:
        question = extract_question(response)
        if question is not None:
            output.append(question)
    return output


def require_exact_mapping(path: Path) -> None:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if (
        mapping.get("schema_version")
        != "ccb-v1-v3-actionable-exact-mapping-v1"
        or mapping.get("summary", {}).get("shared_tasks_exact") != 328
        or mapping.get("summary", {}).get("unmapped_v3_tasks") != 0
        or mapping.get("execution_policy", {}).get(
            "requires_separate_generation_task_ids"
        )
        != []
    ):
        raise av.EvaluationError(
            "V1/V3 mapping requires separate baseline generation"
        )


def load_units(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, set[str]],
    dict[str, Any],
]:
    mapping = json.loads(
        args.v1_v3_mapping.read_text(encoding="utf-8")
    )
    task_mappings = av.index_unique(
        mapping["task_mappings"], "v3_task_id", "V1/V3 task mappings"
    )
    v3_rows = av.index_unique(
        util.read_jsonl(args.v3_actionable), "task_id", "V3 actionable"
    )
    v1_rows = av.index_unique(
        util.read_jsonl(args.v1_variants), "task_id", "V1 variants"
    )
    legacy_v1_rows = av.index_unique(
        util.read_jsonl(args.legacy_v1_variants),
        "task_id",
        "legacy V1 variants",
    )
    vanilla = av.index_unique(
        util.read_jsonl(args.vanilla_results),
        "task_id",
        "Vanilla LLM results",
    )
    ask = av.index_unique(
        util.read_jsonl(args.ask_or_assume_results),
        "task_id",
        "Ask-or-Assume results",
    )
    clarify = av.index_unique(
        util.read_jsonl(args.clarifygpt_results),
        "task_id",
        "ClarifyGPT results",
    )
    if (
        len(v3_rows) != 328
        or len(v1_rows) != 419
        or len(legacy_v1_rows) != 419
        or len(vanilla) != 419
        or len(ask) != 419
        or len(clarify) != 419
    ):
        raise av.EvaluationError("benchmark or baseline task census mismatch")
    if set(vanilla) != set(v1_rows) or set(ask) != set(v1_rows):
        raise av.EvaluationError("interactive baseline task coverage mismatch")
    if set(clarify) != set(v1_rows):
        raise av.EvaluationError("ClarifyGPT task coverage mismatch")
    if set(legacy_v1_rows) != set(v1_rows):
        raise av.EvaluationError("legacy/current V1 task coverage mismatch")

    memberships: dict[str, set[str]] = defaultdict(set)
    raw_counts: Counter[str] = Counter()
    unique_by_method: dict[str, set[str]] = defaultdict(set)
    task_question_counts: dict[str, Counter[str]] = {}
    units: list[dict[str, Any]] = []
    evaluation_units: list[dict[str, Any]] = []
    seen_gold: set[str] = set()
    reused_gold = 0
    evaluated_gold = 0
    changed_tasks = 0
    matcher_tasks = 0
    legacy_v1_question_rewrites = 0

    for task_id in sorted(
        v3_rows,
        key=lambda value: int(value.removeprefix("task_")),
    ):
        v3 = v3_rows[task_id]
        v1 = v1_rows[task_id]
        legacy_v1 = legacy_v1_rows[task_id]
        if (
            v3.get("schema_version") != "ClarifyCodeBench-v3-actionable"
            or not isinstance(v3.get("qa_pairs"), list)
            or len(v3["qa_pairs"]) != v3.get("k")
        ):
            raise av.EvaluationError(f"{task_id}: invalid V3 row")
        requirement = str(v3["modified_content"])
        question_id = str(v3["question_id"])
        if (
            str(v1.get("modified_content")) != requirement
            or str(v1.get("question_id")) != question_id
        ):
            raise av.EvaluationError(f"{task_id}: V1/V3 task drift")
        for name, row in (
            ("Vanilla LLM", vanilla[task_id]),
            ("Ask-or-Assume", ask[task_id]),
        ):
            if (
                row.get("finished") is not True
                or str(row.get("prompt")) != requirement
                or str(row.get("id")) != question_id
            ):
                raise av.EvaluationError(
                    f"{task_id}: invalid {name} frozen result"
                )
        if (
            clarify[task_id].get("status") != "ok"
            or str(clarify[task_id].get("question_id")) != question_id
            or str(clarify[task_id].get("model")) != "gpt-5-mini"
        ):
            raise av.EvaluationError(
                f"{task_id}: invalid ClarifyGPT frozen result"
            )
        task_mapping = task_mappings[task_id]
        shared_questions = {
            str(value["question"])
            for value in task_mapping["shared_gold_exact"]
        }
        new_questions = {
            str(value["question"])
            for value in task_mapping["v3_only_gold_exact"]
        }
        if shared_questions & new_questions:
            raise av.EvaluationError(
                f"{task_id}: overlapping shared/new Gold"
            )
        v3_questions = {
            str(value["question"]) for value in v3["qa_pairs"]
        }
        if shared_questions | new_questions != v3_questions:
            raise av.EvaluationError(
                f"{task_id}: mapping does not partition V3 Gold"
            )
        changed_tasks += int(
            bool(task_mapping["v1_only_gold_exact"])
            or bool(task_mapping["v3_only_gold_exact"])
        )
        matcher_tasks += int(bool(new_questions))

        current_pairs = list(v1["qa_pairs"])
        legacy_pairs = list(legacy_v1["qa_pairs"])
        v1_questions = {
            str(value["question"]) for value in current_pairs
        }
        legacy_questions = {
            str(value["question"]) for value in legacy_pairs
        }
        if (
            str(legacy_v1.get("modified_content")) != requirement
            or str(legacy_v1.get("question_id")) != question_id
            or len(current_pairs) != len(legacy_pairs)
            or any(
                str(current["answer"]) != str(legacy["answer"])
                for current, legacy in zip(current_pairs, legacy_pairs)
            )
        ):
            raise av.EvaluationError(
                f"{task_id}: legacy/current V1 alignment drift"
            )
        legacy_v1_question_rewrites += sum(
            str(current["question"]) != str(legacy["question"])
            for current, legacy in zip(current_pairs, legacy_pairs)
        )
        legacy_prior_hits = {
            "Vanilla LLM": {
                str(value)
                for value in vanilla[task_id].get("hit_questions") or []
            },
            "Ask-or-Assume": {
                str(value)
                for value in ask[task_id].get("hit_questions") or []
            },
        }
        for method, hits in legacy_prior_hits.items():
            if not hits <= legacy_questions:
                raise av.EvaluationError(
                    f"{task_id}: invalid reused {method} hit evidence"
                )
        clarify_prior_hits = {
            str(value)
            for value in (
                clarify[task_id].get("matched_key_questions") or []
            )
        }
        if not clarify_prior_hits <= v1_questions:
            raise av.EvaluationError(
                f"{task_id}: invalid reused ClarifyGPT hit evidence"
            )
        prior_hits_by_current_question = {
            str(current["question"]): {
                "Vanilla LLM": (
                    str(legacy["question"])
                    in legacy_prior_hits["Vanilla LLM"]
                ),
                "Ask-or-Assume": (
                    str(legacy["question"])
                    in legacy_prior_hits["Ask-or-Assume"]
                ),
                "ClarifyGPT": (
                    str(current["question"]) in clarify_prior_hits
                ),
            }
            for current, legacy in zip(current_pairs, legacy_pairs)
        }

        method_questions = {
            "Vanilla LLM": trajectory_questions(vanilla[task_id]),
            "Ask-or-Assume": trajectory_questions(ask[task_id]),
            "ClarifyGPT": [
                str(value).strip()
                for value in clarify[task_id].get("questions") or []
                if str(value).strip()
            ],
        }
        counts = Counter()
        candidates_by_text: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            questions = method_questions[method]
            raw_counts[method] += len(questions)
            counts[method] = len(questions)
            for question in questions:
                digest = hashlib.sha256(
                    f"{task_id}\0{question}".encode("utf-8")
                ).hexdigest()[:24]
                candidate_id = f"baseline::{task_id}::{digest}"
                memberships[candidate_id].add(method)
                unique_by_method[method].add(candidate_id)
                candidates_by_text.setdefault(
                    question,
                    {
                        "candidate_id": candidate_id,
                        "question": question,
                    },
                )
        task_question_counts[task_id] = counts

        item_id = "v_" + task_id
        gold: list[dict[str, Any]] = []
        old_questions = {
            str(value["question"]) for value in v1["qa_pairs"]
        }
        for index, value in enumerate(v3["qa_pairs"], 1):
            gold_id = f"{item_id}::gold_{index:02d}"
            if gold_id in seen_gold:
                raise av.EvaluationError(f"{gold_id}: duplicate Gold ID")
            seen_gold.add(gold_id)
            question = str(value["question"])
            source = (
                "reused_v1_exact"
                if question in shared_questions
                else "new_v3_matcher"
            )
            reused_gold += int(source == "reused_v1_exact")
            evaluated_gold += int(source == "new_v3_matcher")
            gold.append(
                {
                    "gold_id": gold_id,
                    "question": question,
                    "answer": str(value["answer"]),
                    "provenance": str(value.get("provenance")),
                    "exact_v1_question": (
                        question in old_questions
                    ),
                    "evaluation_source": source,
                    "reused_stage_hits": (
                        prior_hits_by_current_question[question]
                        if source == "reused_v1_exact"
                        else None
                    ),
                }
            )
        unit = {
            "schema_version": "ccb-v3-frozen-baseline-unit-v1",
            "item_id": item_id,
            "task_id": task_id,
            "question_id": question_id,
            "requirement": requirement,
            "gold": gold,
            "candidates": sorted(
                candidates_by_text.values(),
                key=lambda row: str(row["candidate_id"]),
            ),
        }
        units.append(unit)
        new_gold = [
            value
            for value in gold
            if value["evaluation_source"] == "new_v3_matcher"
        ]
        if new_gold:
            evaluation_units.append({**unit, "gold": new_gold})

    if (
        len(seen_gold) != 463
        or reused_gold != 416
        or evaluated_gold != 47
    ):
        raise av.EvaluationError("V3 Gold census mismatch")
    all_candidate_ids = {
        str(candidate["candidate_id"])
        for unit in units
        for candidate in unit["candidates"]
    }
    if set(memberships) != all_candidate_ids:
        raise av.EvaluationError("baseline candidate membership drift")
    overlap = {
        "v3_tasks": len(units),
        "v3_gold": len(seen_gold),
        "source_problem_clusters": len(
            {str(unit["question_id"]) for unit in units}
        ),
        "unchanged_gold_reused_from_v1": reused_gold,
        "new_or_revised_v3_gold_evaluated": evaluated_gold,
        "changed_tasks": changed_tasks,
        "matcher_task_count": matcher_tasks,
        "unchanged_task_count": len(units) - changed_tasks,
        "v3_only_task_count": 0,
        "legacy_v1_question_rewrites_with_exact_answers": (
            legacy_v1_question_rewrites
        ),
        "combined_unique_candidate_count": len(all_candidate_ids),
        "evaluated_unique_candidate_count": len(
            {
                str(candidate["candidate_id"])
                for unit in evaluation_units
                for candidate in unit["candidates"]
            }
        ),
        "raw_question_counts": dict(raw_counts),
        "unique_question_counts": {
            method: len(unique_by_method[method]) for method in METHODS
        },
        "tasks_with_questions": {
            method: sum(
                task_question_counts[task_id][method] > 0
                for task_id in task_question_counts
            )
            for method in METHODS
        },
        "incremental_evaluation_note": (
            "Exact shared V1/V3 Gold outcomes are reused from each "
            "baseline's frozen V1 hit evidence. Matcher calls cover only "
            "the 47 V3-only or revised Gold questions."
        ),
    }
    return units, evaluation_units, memberships, overlap


def stage_metrics(
    *,
    gold_results: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    stage: str,
    candidate_count: int,
) -> dict[str, Any]:
    by_item: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in gold_results:
        by_item[str(row["item_id"])].append(row)
    hits = sum(bool(row["stage_hits"][stage]) for row in gold_results)
    all_gold = sum(
        all(bool(row["stage_hits"][stage]) for row in rows)
        for rows in by_item.values()
    )
    any_gold = sum(
        any(bool(row["stage_hits"][stage]) for row in rows)
        for rows in by_item.values()
    )
    return {
        "candidate_count": candidate_count,
        "gold_hits": hits,
        "gold_total": len(gold_results),
        "gold_recall": hits / len(gold_results),
        "all_gold_tasks": all_gold,
        "any_gold_tasks": any_gold,
        "task_total": len(units),
        "all_gold_task_rate": all_gold / len(units),
        "any_gold_task_rate": any_gold / len(units),
    }


def render_report(output: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen baselines on ClarifyCodeBench-v3-actionable",
        "",
        (
            "All baseline generations are frozen gpt-5-mini artifacts. "
            "The 416 exact shared V1/V3 Gold outcomes are reused; matcher "
            "calls cover only the 47 new or revised V3 Gold questions."
        ),
        "",
        "## Gold recall",
        "",
        (
            "| Method | Gold recall | Any-Gold tasks | All-Gold tasks "
            "| Unique questions |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for stage in output["stage_order"]:
        row = output["stages"][stage]
        lines.append(
            f"| {stage} | {row['gold_hits']}/{row['gold_total']} "
            f"({100 * row['gold_recall']:.2f}%) "
            f"| {row['any_gold_tasks']}/{row['task_total']} "
            f"({100 * row['any_gold_task_rate']:.2f}%) "
            f"| {row['all_gold_tasks']}/{row['task_total']} "
            f"({100 * row['all_gold_task_rate']:.2f}%) "
            f"| {row['candidate_count']} |"
        )
    lines.extend(["", "## Paired cluster-bootstrap deltas", ""])
    for name, row in output["paired_deltas"].items():
        lines.append(
            f"- {name}: {100 * row['point_delta']:+.2f} pp, "
            f"95% cluster-bootstrap CI "
            f"[{100 * row['ci95'][0]:+.2f}, "
            f"{100 * row['ci95'][1]:+.2f}]"
        )
    agreement = output["evaluator_agreement"]
    lines.extend(
        [
            "",
            "## Evaluator agreement",
            "",
            (
                f"- Fleiss' kappa: {agreement['fleiss_kappa']:.4f}; "
                f"unanimous pairs: {agreement['unanimous_pairs']}/"
                f"{agreement['pair_count']}."
            ),
            "",
            (
                "ClarifyCodeBench-v3-actionable remains a provisional "
                "correction snapshot; these are development results."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def finalize(
    *,
    args: argparse.Namespace,
    units: Sequence[Mapping[str, Any]],
    memberships: Mapping[str, set[str]],
    overlap: Mapping[str, Any],
    calls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    vote_counts: Counter[tuple[str, str]] = Counter()
    pair_universe: set[tuple[str, str]] = set()
    for unit in units:
        candidate_ids = {
            str(row["candidate_id"]) for row in unit["candidates"]
        }
        gold_ids = {
            str(row["gold_id"])
            for row in unit["gold"]
            if row["evaluation_source"] == "new_v3_matcher"
        }
        pair_universe.update(
            (candidate_id, gold_id)
            for candidate_id in candidate_ids
            for gold_id in gold_ids
        )
    for call in calls.values():
        for row in call["matches"]:
            for gold_id in row["gold_ids"]:
                pair = (str(row["candidate_id"]), str(gold_id))
                if pair not in pair_universe:
                    raise av.EvaluationError("matcher returned invalid pair")
                vote_counts[pair] += 1

    threshold = args.replicates // 2 + 1
    match_rows: list[dict[str, Any]] = []
    matches_by_gold: dict[str, set[str]] = defaultdict(set)
    for candidate_id, gold_id in sorted(pair_universe):
        votes = vote_counts[(candidate_id, gold_id)]
        if votes >= threshold:
            matches_by_gold[gold_id].add(candidate_id)
        match_rows.append(
            {
                "schema_version": SCHEMA_MATCH,
                "candidate_id": candidate_id,
                "gold_id": gold_id,
                "votes": votes,
                "matched": votes >= threshold,
            }
        )
    util.atomic_bytes(
        args.output_dir / "candidate_gold_pair_votes.jsonl",
        util.jsonl_bytes(match_rows),
    )

    av_gold = av.index_unique(
        util.read_jsonl(args.av_v3_gold_results),
        "gold_id",
        "frozen AV V3 Gold results",
    )
    av_summary = json.loads(
        args.av_v3_summary.read_text(encoding="utf-8")
    )
    av_stages = list(av_summary["stage_order"])
    gold_results: list[dict[str, Any]] = []
    for unit in units:
        for gold in unit["gold"]:
            gold_id = str(gold["gold_id"])
            if gold_id not in av_gold:
                raise av.EvaluationError(f"{gold_id}: absent from AV results")
            prior = av_gold[gold_id]
            if (
                str(prior["question"]) != str(gold["question"])
                or str(prior["answer"]) != str(gold["answer"])
                or str(prior["question_id"]) != str(unit["question_id"])
            ):
                raise av.EvaluationError(f"{gold_id}: AV Gold drift")
            matched = matches_by_gold[gold_id]
            if gold["evaluation_source"] == "reused_v1_exact":
                baseline_hits = dict(gold["reused_stage_hits"])
            else:
                baseline_hits = {
                    method: any(
                        method in memberships[candidate_id]
                        for candidate_id in matched
                    )
                    for method in METHODS
                }
            stage_hits = {
                **baseline_hits,
                **{
                    stage: bool(prior["stage_hits"][stage])
                    for stage in av_stages
                },
            }
            gold_results.append(
                {
                    "schema_version": (
                        "ccb-v3-frozen-baseline-gold-result-v1"
                    ),
                    "gold_id": gold_id,
                    "item_id": unit["item_id"],
                    "question_id": unit["question_id"],
                    "question": gold["question"],
                    "answer": gold["answer"],
                    "provenance": gold["provenance"],
                    "matched_candidate_ids": sorted(matched),
                    "baseline_evaluation_source": gold[
                        "evaluation_source"
                    ],
                    "baseline_stage_hits": baseline_hits,
                    "stage_hits": stage_hits,
                }
            )
    if len(gold_results) != 463 or len(av_gold) != 463:
        raise av.EvaluationError("combined Gold result census mismatch")
    util.atomic_bytes(
        args.output_dir / "gold_results.jsonl",
        util.jsonl_bytes(gold_results),
    )

    unique_counts = overlap["unique_question_counts"]
    stages = {
        method: stage_metrics(
            gold_results=gold_results,
            units=units,
            stage=method,
            candidate_count=int(unique_counts[method]),
        )
        for method in METHODS
    }
    for stage in av_stages:
        stages[stage] = dict(av_summary["stages"][stage])
        computed_hits = sum(
            bool(row["stage_hits"][stage]) for row in gold_results
        )
        if computed_hits != stages[stage]["gold_hits"]:
            raise av.EvaluationError(f"{stage}: frozen AV metric drift")

    comparisons = [
        ("Vanilla_to_Ask-or-Assume", "Vanilla LLM", "Ask-or-Assume"),
        ("Vanilla_to_ClarifyGPT", "Vanilla LLM", "ClarifyGPT"),
        ("Ask-or-Assume_to_ClarifyGPT", "Ask-or-Assume", "ClarifyGPT"),
        ("ClarifyGPT_to_QG", "ClarifyGPT", "QG"),
    ]
    paired_deltas = {
        name: av.bootstrap_delta(
            gold_rows=gold_results,
            left=left,
            right=right,
            samples=args.bootstrap_samples,
            seed=args.seed + index,
        )
        for index, (name, left, right) in enumerate(comparisons)
    }
    pair_votes = [vote_counts[pair] for pair in sorted(pair_universe)]
    if not pair_votes:
        raise av.EvaluationError("baseline pair-vote census is empty")
    output = {
        "schema_version": "ccb-v3-frozen-baseline-summary-v1",
        "completed_at": util.utc_now(),
        "benchmark_status": (
            "provisional correction snapshot; development evidence only"
        ),
        "baseline_backbone": "gpt-5-mini",
        "no_baseline_generation_rerun": True,
        "evaluation_protocol": (
            "three-replicate same-decision candidate-to-Gold matching"
        ),
        "overlap": dict(overlap),
        "stage_order": [*METHODS, *av_stages],
        "stages": stages,
        "paired_deltas": paired_deltas,
        "evaluator_agreement": {
            "replicates": args.replicates,
            "pair_count": len(pair_votes),
            "unanimous_pairs": sum(
                votes in {0, args.replicates} for votes in pair_votes
            ),
            "split_pairs": sum(
                votes not in {0, args.replicates} for votes in pair_votes
            ),
            "positive_majority_pairs": sum(
                votes >= threshold for votes in pair_votes
            ),
            "fleiss_kappa": av.fleiss_kappa(
                pair_votes, args.replicates
            ),
        },
        "hashes": {
            "v3_actionable": av.sha256_file(args.v3_actionable),
            "v1_variants": av.sha256_file(args.v1_variants),
            "legacy_v1_variants": av.sha256_file(
                args.legacy_v1_variants
            ),
            "v1_v3_mapping": av.sha256_file(args.v1_v3_mapping),
            "vanilla_results": av.sha256_file(args.vanilla_results),
            "ask_or_assume_results": av.sha256_file(
                args.ask_or_assume_results
            ),
            "clarifygpt_results": av.sha256_file(
                args.clarifygpt_results
            ),
            "av_v3_summary": av.sha256_file(args.av_v3_summary),
            "av_v3_gold_results": av.sha256_file(
                args.av_v3_gold_results
            ),
            "calls": av.sha256_file(args.output_dir / "calls.jsonl"),
            "pair_votes": av.sha256_file(
                args.output_dir / "candidate_gold_pair_votes.jsonl"
            ),
            "gold_results": av.sha256_file(
                args.output_dir / "gold_results.jsonl"
            ),
        },
    }
    util.atomic_bytes(
        args.output_dir / "summary.json", util.json_bytes(output)
    )
    util.atomic_bytes(
        args.output_dir / "summary.md",
        render_report(output).encode("utf-8"),
    )
    return output


def execute(args: argparse.Namespace) -> dict[str, Any] | None:
    require_exact_mapping(args.v1_v3_mapping)
    units, evaluation_units, memberships, overlap = load_units(args)
    batches = av.build_batches(
        evaluation_units, args.candidate_batch_size
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    units_path = args.output_dir / "units.jsonl"
    util.write_immutable(units_path, util.jsonl_bytes(units))
    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "method": (
            "three-replicate same-decision candidate-to-Gold matching"
        ),
        "model": args.model,
        "temperature": 0.0,
        "replicates": args.replicates,
        "workers": args.workers,
        "candidate_batch_size": args.candidate_batch_size,
        "batch_count": len(batches),
        "expected_call_count": len(batches) * args.replicates,
        "task_count": len(units),
        "gold_count": sum(len(unit["gold"]) for unit in units),
        "matcher_gold_count": sum(
            len(unit["gold"]) for unit in evaluation_units
        ),
        "matcher_task_count": len(evaluation_units),
        "combined_unique_candidate_count": sum(
            len(unit["candidates"]) for unit in units
        ),
        "system_prompt": av.SYSTEM_PROMPT,
        "system_prompt_sha256": util.stable_json_sha256(
            av.SYSTEM_PROMPT
        ),
        "gate_visibility_to_evaluator": False,
        "baseline_backbone": "gpt-5-mini",
        "no_baseline_generation_rerun": True,
        "incremental_v1_reuse": True,
        "input_sha256": {
            "v3_actionable": av.sha256_file(args.v3_actionable),
            "v1_variants": av.sha256_file(args.v1_variants),
            "legacy_v1_variants": av.sha256_file(
                args.legacy_v1_variants
            ),
            "v1_v3_mapping": av.sha256_file(args.v1_v3_mapping),
            "vanilla_results": av.sha256_file(args.vanilla_results),
            "ask_or_assume_results": av.sha256_file(
                args.ask_or_assume_results
            ),
            "clarifygpt_results": av.sha256_file(
                args.clarifygpt_results
            ),
            "av_v3_summary": av.sha256_file(args.av_v3_summary),
            "av_v3_gold_results": av.sha256_file(
                args.av_v3_gold_results
            ),
        },
        "units_sha256": av.sha256_file(units_path),
        "code_sha256": av.sha256_file(Path(__file__).resolve()),
        "matcher_code_sha256": av.sha256_file(
            Path(av.__file__).resolve()
        ),
        "overlap": overlap,
    }
    manifest_path = args.output_dir / "manifest.json"
    util.write_immutable(manifest_path, util.json_bytes(manifest))
    if args.prepare_only:
        return None
    if not args.confirm_live:
        raise av.EvaluationError("live evaluation requires --confirm-live")

    complete_path = args.output_dir / "COMPLETE.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            complete.get("manifest_sha256")
            != av.sha256_file(manifest_path)
            or complete.get("summary_sha256")
            != av.sha256_file(args.output_dir / "summary.json")
        ):
            raise av.EvaluationError("existing completion failed integrity")
        return json.loads(
            (args.output_dir / "summary.json").read_text(encoding="utf-8")
        )

    base_url = os.environ.get("CONTRA_BASE_URL")
    api_key = os.environ.get("CONTRA_API_KEY")
    if not base_url or not api_key:
        raise av.EvaluationError("set CONTRA_BASE_URL and CONTRA_API_KEY")
    import openai

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=600.0,
        max_retries=0,
    )
    calls_path = args.output_dir / "calls.jsonl"
    if not calls_path.exists():
        util.atomic_bytes(calls_path, b"")
    util.repair_torn_tail(calls_path)
    existing = av.index_unique(
        util.read_jsonl(calls_path), "call_key", "call journal"
    )
    tasks = [
        (batch, replicate)
        for batch in batches
        for replicate in range(args.replicates)
    ]
    pending = [
        (batch, replicate)
        for batch, replicate in tasks
        if f"{batch['batch_id']}::replicate_{replicate}" not in existing
    ]
    lock = threading.Lock()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                av.call_unit,
                unit=batch,
                replicate=replicate,
                client=client,
                model=args.model,
            ): (str(batch["batch_id"]), replicate)
            for batch, replicate in pending
        }
        for future in as_completed(futures):
            batch_id, replicate = futures[future]
            try:
                row = future.result()
                with lock:
                    util.append_jsonl(calls_path, row)
            except Exception as error:
                failures.append(
                    f"{batch_id}/replicate_{replicate}: "
                    f"{type(error).__name__}: {error}"
                )
    if failures:
        util.atomic_bytes(
            args.output_dir / "FAILURES.json",
            util.json_bytes({"failures": failures}),
        )
        raise av.EvaluationError(
            f"{len(failures)} evaluator calls failed"
        )

    util.repair_torn_tail(calls_path)
    calls = av.index_unique(
        util.read_jsonl(calls_path), "call_key", "call journal"
    )
    expected_keys = {
        f"{batch['batch_id']}::replicate_{replicate}"
        for batch, replicate in tasks
    }
    if set(calls) != expected_keys:
        raise av.EvaluationError("evaluator call census mismatch")
    output = finalize(
        args=args,
        units=units,
        memberships=memberships,
        overlap=overlap,
        calls=calls,
    )
    complete = {
        "schema_version": "ccb-v3-frozen-baseline-complete-v1",
        "completed_at": util.utc_now(),
        "manifest_sha256": av.sha256_file(manifest_path),
        "summary_sha256": av.sha256_file(
            args.output_dir / "summary.json"
        ),
        "calls_sha256": av.sha256_file(calls_path),
        "call_count": len(calls),
    }
    util.atomic_bytes(complete_path, util.json_bytes(complete))
    return output


def main() -> None:
    args = parse_args()
    output = execute(args)
    if output is not None:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
