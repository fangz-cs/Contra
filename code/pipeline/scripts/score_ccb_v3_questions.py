#!/usr/bin/env python3
"""Compare ordered-question policies on ClarifyCodeBench-v3.

Final code is ignored. Each run is normalized to its first five clarification
turns, followed by one logical terminal turn, matching the official six-turn
interaction budget.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from forkcast.ccb_v3_harness import HarnessError, index_tasks, read_jsonl
from forkcast.ccb_v3_scoring import (
    finished_by_task,
    loglinear_d_prime,
    per_task,
    summarize,
)


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    return label, Path(raw_path)


def parse_control_asks(value: str) -> tuple[str, int]:
    label, separator, raw_count = value.partition("=")
    if not separator or not label:
        raise argparse.ArgumentTypeError(
            "--control-asks must be LABEL=COUNT"
        )
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--control-asks COUNT must be an integer"
        ) from exc
    return label, count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data_v3/ClarifyCodeBench-v3-actionable.jsonl"),
    )
    parser.add_argument(
        "--release",
        type=Path,
        required=True,
        help="official clarifycodebench_release checkout",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="repeatable LABEL=PATH",
    )
    parser.add_argument("--question-budget", type=int, default=5)
    parser.add_argument(
        "--allow-extra-run-tasks",
        action="store_true",
        help=(
            "accept a diagnostic run containing additional tasks, while "
            "scoring only the 328 actionable task IDs"
        ),
    )
    parser.add_argument(
        "--control-asks",
        action="append",
        type=parse_control_asks,
        default=[],
        help="repeatable LABEL=COUNT for the independent complete controls",
    )
    parser.add_argument("--control-total", type=int, default=80)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.question_budget < 0:
        raise SystemExit("question-budget must be non-negative")
    sys.path.insert(0, str(args.release.resolve()))
    from clarifycodebench import metrics as official_metrics  # noqa: PLC0415

    tasks = index_tasks(read_jsonl(args.data))
    expected_gold = sum(len(task.get("qa_pairs") or []) for task in tasks.values())
    expected_k0 = sum(
        not (task.get("qa_pairs") or []) for task in tasks.values()
    )
    if len(tasks) != 328 or expected_gold != 463 or expected_k0 != 0:
        raise HarnessError("unexpected V3 actionable dataset census")
    if args.control_total != 80:
        raise HarnessError("the frozen complete-control denominator is 80")

    control_asks: dict[str, int] = {}
    for label, count in args.control_asks:
        if label in control_asks:
            raise HarnessError(f"duplicate control count for {label}")
        if not 0 <= count <= args.control_total:
            raise HarnessError(f"{label}: invalid complete-control count")
        control_asks[label] = count

    output: dict[str, Any] = {
        "schema_version": "ccb-v3-question-comparison-v2",
        "evaluation_scope": "clarification_only",
        "actionable_tasks": 328,
        "gold_questions": 463,
        "complete_control_tasks": args.control_total,
        "question_budget": args.question_budget,
        "logical_terminal_turn": True,
        "runs": {},
    }
    seen_labels: set[str] = set()
    for label, path in args.run:
        if label in seen_labels:
            raise HarnessError(f"duplicate run label: {label}")
        seen_labels.add(label)
        records = finished_by_task(read_jsonl(path))
        missing = sorted(set(tasks) - set(records))
        extra = sorted(set(records) - set(tasks))
        if missing or (extra and not args.allow_extra_run_tasks):
            raise HarnessError(
                f"{label}: task coverage mismatch "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        task_rows = [
            per_task(
                record=records[task_id],
                task=task,
                official_metrics=official_metrics,
                question_budget=args.question_budget,
            )
            for task_id, task in tasks.items()
        ]
        summary = summarize(task_rows)
        summary["extra_run_tasks_ignored"] = len(extra)
        if label in control_asks:
            false_alarm_count = control_asks[label]
            summary.update(
                {
                    "F_count": false_alarm_count,
                    "F": false_alarm_count / args.control_total,
                    "d_prime": loglinear_d_prime(
                        hit_count=int(summary["H_count"]),
                        hit_total=328,
                        false_alarm_count=false_alarm_count,
                        false_alarm_total=args.control_total,
                    ),
                }
            )
        output["runs"][label] = summary
    unknown_control_labels = sorted(set(control_asks) - seen_labels)
    if unknown_control_labels:
        raise HarnessError(
            f"control counts have no matching run: {unknown_control_labels}"
        )

    print(
        f"{'method':24} {'KQR':>7} {'TKQR':>7} {'ORA':>7} "
        f"{'H':>7} {'F':>7} {'d-prime':>8} {'questions':>10}"
    )
    for label, summary in output["runs"].items():
        f_value = summary.get("F")
        d_prime = summary.get("d_prime")
        prefix = (
            f"{label:24} {summary['KQR_macro']:7.3f} "
            f"{summary['TKQR']:7.3f} {summary['ORA']:7.3f} "
            f"{summary['H']:7.3f} "
        )
        f_text = f"{f_value:7.3f}" if f_value is not None else f"{'-':>7}"
        d_text = (
            f"{d_prime:8.3f}" if d_prime is not None else f"{'-':>8}"
        )
        print(
            f"{prefix}{f_text} {d_text} "
            f"{summary['avg_question_turns']:10.3f}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
