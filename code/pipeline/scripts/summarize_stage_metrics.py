#!/usr/bin/env python3
"""Combine Gold-question Recall@5 and source-control task FPR by stage."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


class SummaryError(RuntimeError):
    pass


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SummaryError(f"{path}: expected object")
    return value


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


def summarize(
    *,
    plan_report_path: Path,
    metrics_path: Path,
    matcher_model: str = "gpt-4o",
) -> dict[str, Any]:
    report = read_object(plan_report_path)
    metrics = read_object(metrics_path)
    denominators = report.get("denominators") or {}
    if (
        denominators.get("gold_questions") != 463
        or denominators.get("complete_source_control_tasks") != 80
        or metrics.get("question_budget") != 5
    ):
        raise SummaryError("stage metric denominator mismatch")
    metric_runs = metrics.get("runs") or {}
    output_stages: dict[str, dict[str, Any]] = {}
    for stage, plan_stage in (report.get("stages") or {}).items():
        label = plan_stage["method_label"]
        values = metric_runs.get(label)
        if (
            not isinstance(values, dict)
            or values.get("gold") != 463
            or values.get("tasks") != 419
        ):
            raise SummaryError(f"{stage}: missing stage metric")
        hits = int(values.get("hits") or 0)
        retained = int(plan_stage["source_control_tasks_retained"])
        output_stages[stage] = {
            "gold_question_hits_at_5": hits,
            "gold_question_recall_at_5": hits / 463,
            "source_control_tasks_retained": retained,
            "source_control_task_fpr": retained / 80,
            "variant_questions_in_plan": int(
                plan_stage["question_count_after_exact_deduplication"]
            ),
            "method_label": label,
        }
    expected = {"qg", "agent1", "agent2", "self_bv", "final"}
    if set(output_stages) != expected:
        raise SummaryError("stage metric coverage mismatch")
    return {
        "schema_version": "ccb-v3-stage-recall-fpr-v1",
        "recall": {
            "unit": "Gold clarification question",
            "denominator": 463,
            "question_budget_per_task": 5,
            "matcher": matcher_model,
            "consistency": 3,
        },
        "fpr": {
            "unit": "complete-source control task",
            "denominator": 80,
            "positive_rule": "at least one candidate survives the stage",
        },
        "stages": output_stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--matcher-model", default="gpt-4o")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = summarize(
        plan_report_path=args.plan_report,
        metrics_path=args.metrics,
        matcher_model=args.matcher_model,
    )
    atomic_json(args.out, output)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
