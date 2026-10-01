#!/usr/bin/env python3
"""Compare selector prompt variants on one paired task set."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = str(row["task_id"])
        if task_id in rows:
            raise ValueError(f"{path}: duplicate task {task_id}")
        if row.get("finished") is not True:
            raise ValueError(f"{path}: unfinished task {task_id}")
        rows[task_id] = row
    return rows


def load_metrics(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "selector_eval_ccb_metrics", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import metrics module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(rows: list[dict[str, Any]], metrics: Any) -> dict[str, Any]:
    true_positives = sum(
        sum(int(value) == 1 for value in row["hit_dict"].values())
        for row in rows
    )
    questions = sum(len(row["asked_questions"]) for row in rows)
    gold_questions = sum(int(row["key_question_num"]) for row in rows)
    precision = true_positives / questions if questions else 0.0
    recall = (
        true_positives / gold_questions if gold_questions else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    tkqr = sum(
        metrics.score_record(row)["tkqr"] for row in rows
    ) / len(rows)
    return {
        "tasks": len(rows),
        "true_positive_questions": true_positives,
        "questions_asked": questions,
        "gold_questions": gold_questions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tkqr": tkqr,
        "average_questions": questions / len(rows),
        "zero_question_tasks": sum(
            not row["asked_questions"] for row in rows
        ),
        "exact_code_rate": sum(
            row.get("final_response_status") == "exact_code_block"
            for row in rows
        )
        / len(rows),
        "stop_reasons": dict(
            Counter(row["selector_stop_reason"] for row in rows)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--subset-receipt", type=Path, required=True)
    parser.add_argument("--metrics-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-result-superset",
        action="store_true",
        help="Score only the requested task IDs from larger result files.",
    )
    parser.add_argument(
        "--precision-tolerance", type=float, default=0.03
    )
    parser.add_argument("--minimum-f1-gain", type=float, default=0.03)
    args = parser.parse_args()
    if not args.variant:
        parser.error("at least one --variant is required")

    receipt = json.loads(
        args.subset_receipt.read_text(encoding="utf-8")
    )
    task_ids = [str(task_id) for task_id in receipt["task_ids"]]
    task_id_set = set(task_ids)
    metrics = load_metrics(args.metrics_module)

    baseline_all = load_rows(args.baseline)
    if not task_id_set <= set(baseline_all):
        raise ValueError("baseline does not cover the development subset")
    baseline = score(
        [baseline_all[task_id] for task_id in task_ids], metrics
    )

    variants: dict[str, dict[str, Any]] = {}
    for item in args.variant:
        if "=" not in item:
            parser.error(f"invalid --variant value: {item}")
        name, raw_path = item.split("=", 1)
        rows = load_rows(Path(raw_path))
        if args.allow_result_superset:
            valid_scope = task_id_set <= set(rows)
        else:
            valid_scope = set(rows) == task_id_set
        if not valid_scope:
            raise ValueError(
                f"{name}: result task IDs do not cover the required scope"
            )
        result = score(
            [rows[task_id] for task_id in task_ids],
            metrics,
        )
        result["delta"] = {
            key: result[key] - baseline[key]
            for key in ("precision", "recall", "f1", "tkqr")
        }
        result["promotable"] = (
            result["recall"] > baseline["recall"]
            and result["precision"]
            >= baseline["precision"] - args.precision_tolerance
            and result["f1"]
            >= baseline["f1"] + args.minimum_f1_gain
            and result["tkqr"] >= baseline["tkqr"]
        )
        variants[name] = result

    promotable = [
        name
        for name, result in variants.items()
        if result["promotable"]
    ]
    ranking = sorted(
        variants,
        key=lambda name: (
            variants[name]["promotable"],
            variants[name]["f1"],
            variants[name]["tkqr"],
            variants[name]["precision"],
        ),
        reverse=True,
    )
    report = {
        "schema_version": "dialogue-selector-prompt-comparison-v1",
        "selection_uses_gold": True,
        "task_ids": task_ids,
        "thresholds": {
            "precision_tolerance": args.precision_tolerance,
            "minimum_f1_gain": args.minimum_f1_gain,
            "recall_must_improve": True,
            "tkqr_must_not_decline": True,
        },
        "baseline": baseline,
        "variants": variants,
        "ranking": ranking,
        "selected": ranking[0] if promotable else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
