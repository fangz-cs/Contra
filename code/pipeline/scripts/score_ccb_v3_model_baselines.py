#!/usr/bin/env python3
"""Score model-specific frozen baselines under the formal V3 contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import score_ccb_v3_frozen_replay as replay
from forkcast.ccb_v3_harness import (
    HarnessError,
    index_tasks,
    load_plan,
    read_jsonl,
    sha256_bytes,
)
from forkcast.ccb_v3_scoring import loglinear_d_prime


METHODS = ("Vanilla LLM", "Ask-or-Assume", "ClarifyGPT")


def protocol_questions(row: Mapping[str, Any]) -> list[str]:
    responses = list(row.get("response_contents") or [])
    hit_dict = row.get("hit_dict") or {}
    output: list[str] = []
    for index, content in enumerate(responses):
        if str(index) not in hit_dict:
            continue
        question = replay.extract_question(content)
        if question is None:
            question = str(content).strip()
        if not question:
            raise HarnessError("empty protocol question")
        output.append(question)
    if len(output) != len(hit_dict):
        raise HarnessError("protocol question trajectory is not aligned")
    return output


def clarifygpt_questions(row: Mapping[str, Any]) -> list[str]:
    questions = [
        str(value).strip()
        for value in row.get("questions") or []
        if str(value).strip()
    ]
    if row.get("n_questions") != len(questions):
        raise HarnessError("ClarifyGPT question trajectory is not aligned")
    return questions


def actionable_records(
    path: Path,
    *,
    tasks: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    records = replay.index_unique(read_jsonl(path), "task_id", label)
    if not set(tasks) <= set(records):
        raise HarnessError(f"{label}: actionable task coverage drift")
    return {task_id: records[task_id] for task_id in tasks}


def load_controls(path: Path) -> int:
    value = json.loads(path.read_text(encoding="utf-8"))
    false_alarms = value.get("false_alarm_count")
    if (
        value.get("complete_control_tasks") != 80
        or not isinstance(false_alarms, int)
        or isinstance(false_alarms, bool)
        or not 0 <= false_alarms <= 80
    ):
        raise HarnessError(f"{path}: invalid complete-control result")
    return false_alarms


def assert_plan_alignment(
    *,
    plan: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    extractor: Any,
    question_budget: int,
) -> None:
    for task_id, record in records.items():
        observed = extractor(record)[:question_budget]
        planned = [
            str(row["text"])
            for row in plan["tasks"][task_id]["questions"][:question_budget]
        ]
        if observed != planned:
            raise HarnessError(f"{task_id}: frozen plan trajectory drift")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--v1-variants", type=Path, required=True)
    parser.add_argument("--legacy-v1-variants", type=Path, required=True)
    parser.add_argument("--vanilla-results", type=Path, required=True)
    parser.add_argument("--ask-or-assume-results", type=Path, required=True)
    parser.add_argument("--clarifygpt-results", type=Path, required=True)
    parser.add_argument("--gold-results", type=Path, required=True)
    parser.add_argument("--pair-votes", type=Path, required=True)
    parser.add_argument("--vanilla-plan", type=Path, required=True)
    parser.add_argument("--ask-plan", type=Path, required=True)
    parser.add_argument("--clarifygpt-plan", type=Path, required=True)
    parser.add_argument("--vanilla-controls", type=Path, required=True)
    parser.add_argument("--ask-controls", type=Path, required=True)
    parser.add_argument("--clarifygpt-controls", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--question-budget", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.question_budget != 5:
        raise HarnessError("the formal V3 question budget is five")

    task_rows = read_jsonl(args.data)
    tasks = index_tasks(task_rows)
    if (
        len(tasks) != 328
        or sum(len(row.get("qa_pairs") or []) for row in task_rows) != 463
        or any(not row.get("qa_pairs") for row in task_rows)
    ):
        raise HarnessError("actionable V3 census mismatch")

    current_all = replay.index_unique(
        read_jsonl(args.v1_variants), "task_id", "current V1 variants"
    )
    legacy_all = replay.index_unique(
        read_jsonl(args.legacy_v1_variants),
        "task_id",
        "legacy V1 variants",
    )
    if not set(tasks) <= set(current_all) or not set(tasks) <= set(legacy_all):
        raise HarnessError("V1/V3 task coverage drift")
    current_v1 = {task_id: current_all[task_id] for task_id in tasks}
    legacy_v1 = {task_id: legacy_all[task_id] for task_id in tasks}

    vanilla = actionable_records(
        args.vanilla_results, tasks=tasks, label="Vanilla LLM records"
    )
    ask = actionable_records(
        args.ask_or_assume_results,
        tasks=tasks,
        label="Ask-or-Assume records",
    )
    clarify = actionable_records(
        args.clarifygpt_results,
        tasks=tasks,
        label="ClarifyGPT records",
    )

    plans = {
        "Vanilla LLM": load_plan(
            args.vanilla_plan, dataset_path=args.data, tasks=tasks
        ),
        "Ask-or-Assume": load_plan(
            args.ask_plan, dataset_path=args.data, tasks=tasks
        ),
        "ClarifyGPT": load_plan(
            args.clarifygpt_plan, dataset_path=args.data, tasks=tasks
        ),
    }
    assert_plan_alignment(
        plan=plans["Vanilla LLM"],
        records=vanilla,
        extractor=protocol_questions,
        question_budget=args.question_budget,
    )
    assert_plan_alignment(
        plan=plans["Ask-or-Assume"],
        records=ask,
        extractor=protocol_questions,
        question_budget=args.question_budget,
    )
    assert_plan_alignment(
        plan=plans["ClarifyGPT"],
        records=clarify,
        extractor=clarifygpt_questions,
        question_budget=args.question_budget,
    )

    gold_order, gold_by_question, new_gold = replay.load_gold(
        tasks=tasks,
        gold_path=args.gold_results,
    )
    incremental_matches = replay.load_matched_pairs(
        args.pair_votes,
        allowed_gold=new_gold,
    )
    vanilla_sequences, vanilla_diagnostics = replay.interactive_sequences(
        label="Vanilla LLM",
        records=vanilla,
        current_v1=current_v1,
        legacy_v1=legacy_v1,
        gold_order=gold_order,
        gold_by_question=gold_by_question,
        incremental_matches=incremental_matches,
        question_budget=args.question_budget,
    )
    ask_sequences, ask_diagnostics = replay.interactive_sequences(
        label="Ask-or-Assume",
        records=ask,
        current_v1=current_v1,
        legacy_v1=legacy_v1,
        gold_order=gold_order,
        gold_by_question=gold_by_question,
        incremental_matches=incremental_matches,
        question_budget=args.question_budget,
    )
    clarify_sequences, clarify_diagnostics = replay.clarifygpt_sequences(
        records=clarify,
        current_v1=current_v1,
        gold_order=gold_order,
        gold_by_question=gold_by_question,
        incremental_matches=incremental_matches,
        question_budget=args.question_budget,
    )
    sequences = {
        "Vanilla LLM": vanilla_sequences,
        "Ask-or-Assume": ask_sequences,
        "ClarifyGPT": clarify_sequences,
    }
    controls = {
        "Vanilla LLM": load_controls(args.vanilla_controls),
        "Ask-or-Assume": load_controls(args.ask_controls),
        "ClarifyGPT": load_controls(args.clarifygpt_controls),
    }
    diagnostics = {
        "Vanilla LLM": vanilla_diagnostics,
        "Ask-or-Assume": ask_diagnostics,
        "ClarifyGPT": clarify_diagnostics,
    }

    official_metrics = replay.load_official_metrics(args.release)
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        rows, summary = replay.score_sequences(
            method=method,
            sequences=sequences[method],
            tasks=tasks,
            official_metrics=official_metrics,
            question_budget=args.question_budget,
        )
        false_alarms = controls[method]
        summary.update(
            {
                "F_count": false_alarms,
                "F_total": 80,
                "F": false_alarms / 80,
                "d_prime": loglinear_d_prime(
                    hit_count=int(summary["H_count"]),
                    hit_total=328,
                    false_alarm_count=false_alarms,
                    false_alarm_total=80,
                ),
            }
        )
        all_rows.extend(rows)
        summaries[method] = summary

    paths = {
        "data": args.data,
        "v1_variants": args.v1_variants,
        "legacy_v1_variants": args.legacy_v1_variants,
        "vanilla_results": args.vanilla_results,
        "ask_or_assume_results": args.ask_or_assume_results,
        "clarifygpt_results": args.clarifygpt_results,
        "gold_results": args.gold_results,
        "pair_votes": args.pair_votes,
        "vanilla_plan": args.vanilla_plan,
        "ask_plan": args.ask_plan,
        "clarifygpt_plan": args.clarifygpt_plan,
        "vanilla_controls": args.vanilla_controls,
        "ask_controls": args.ask_controls,
        "clarifygpt_controls": args.clarifygpt_controls,
    }
    output = {
        "schema_version": "ccb-v3-model-baseline-score-v1",
        "model": args.model,
        "question_budget": args.question_budget,
        "logical_terminal_turn": True,
        "task_count": 328,
        "gold_count": 463,
        "control_count": 80,
        "methods": summaries,
        "reconstruction": diagnostics,
        "evidence": {
            "shared_gold": "frozen V1 baseline hit feedback",
            "new_or_revised_gold": (
                "three-replicate GPT-4o incremental pair votes"
            ),
            "final_code_evaluated": False,
        },
        "hashes": {
            label: sha256_bytes(path.read_bytes())
            for label, path in paths.items()
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay.atomic_jsonl(args.out_dir / "task-scores.jsonl", all_rows)
    replay.atomic_json(args.out_dir / "summary.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
