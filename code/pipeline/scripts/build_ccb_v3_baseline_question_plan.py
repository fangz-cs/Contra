#!/usr/bin/env python3
"""Convert frozen baseline trajectories into a V3 actionable question plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from forkcast.ccb_v3_harness import (
    PLAN_SCHEMA,
    HarnessError,
    index_tasks,
    read_jsonl,
    sha256_bytes,
    sha256_text,
)


QUESTION_PATTERN = re.compile(
    r"^\s*\[QUESTION\]\s*(.*?)\s*\[/QUESTION\]\s*$",
    re.DOTALL,
)


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


def extract_question(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessError("baseline emitted an empty question")
    match = QUESTION_PATTERN.fullmatch(value)
    question = match.group(1).strip() if match else value.strip()
    if not question:
        raise HarnessError("baseline emitted an empty question")
    return question


def index_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    record_format: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        healthy = (
            row.get("finished") is True
            if record_format == "protocol"
            else row.get("status") == "ok"
        )
        if not healthy:
            continue
        task_id = row.get("task_id")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in output
        ):
            raise HarnessError(f"{label}: invalid or duplicate healthy task")
        output[task_id] = row
    return output


def protocol_questions(row: Mapping[str, Any]) -> list[str]:
    responses = row.get("response_contents")
    hit_dict = row.get("hit_dict")
    if not isinstance(responses, list) or not isinstance(hit_dict, dict):
        raise HarnessError("protocol record has no question trajectory")
    questions: list[str] = []
    for index, response in enumerate(responses):
        if str(index) not in hit_dict:
            continue
        questions.append(extract_question(response))
    if len(questions) != len(hit_dict):
        raise HarnessError("protocol question trajectory is not index-aligned")
    return questions


def clarifygpt_questions(row: Mapping[str, Any]) -> list[str]:
    values = row.get("questions")
    if not isinstance(values, list):
        raise HarnessError("ClarifyGPT record has no questions list")
    questions = [extract_question(value) for value in values]
    if (
        row.get("n_questions") != len(questions)
        or row.get("asked") is not bool(questions)
    ):
        raise HarnessError("ClarifyGPT question-count drift")
    return questions


def questions_for(
    row: Mapping[str, Any], *, record_format: str
) -> list[str]:
    if record_format == "protocol":
        return protocol_questions(row)
    if record_format == "clarifygpt":
        return clarifygpt_questions(row)
    raise HarnessError(f"unsupported baseline record format: {record_format}")


def candidate_id(
    *, method: str, task_id: str, position: int, question: str
) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:20]
    normalized_method = re.sub(r"[^a-z0-9]+", "-", method.lower()).strip("-")
    return (
        f"baseline::{normalized_method}::{task_id}::{position:02d}::{digest}"
    )


def build(
    *,
    dataset_path: Path,
    underspecified_path: Path,
    complete_path: Path,
    record_format: str,
    method: str,
    model: str,
    binding_underspecified_path: Path | None = None,
    binding_complete_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks = index_tasks(read_jsonl(dataset_path))
    if (
        len(tasks) != 328
        or sum(len(task.get("qa_pairs") or []) for task in tasks.values())
        != 463
        or any(not task.get("qa_pairs") for task in tasks.values())
    ):
        raise HarnessError("unexpected V3 actionable dataset census")

    underspecified = index_records(
        read_jsonl(underspecified_path),
        record_format=record_format,
        label="underspecified records",
    )
    complete = index_records(
        read_jsonl(complete_path),
        record_format=record_format,
        label="complete records",
    )
    binding_underspecified: dict[str, dict[str, Any]] = {}
    binding_complete_by_question: dict[str, dict[str, Any]] = {}
    if record_format == "clarifygpt":
        if (
            binding_underspecified_path is None
            or binding_complete_path is None
        ):
            raise HarnessError(
                "ClarifyGPT conversion requires protocol binding records"
            )
        binding_underspecified = index_records(
            read_jsonl(binding_underspecified_path),
            record_format="protocol",
            label="underspecified binding records",
        )
        binding_complete = index_records(
            read_jsonl(binding_complete_path),
            record_format="protocol",
            label="complete binding records",
        )
        for binding in binding_complete.values():
            question_id = binding.get("id")
            if (
                not isinstance(question_id, str)
                or not question_id
                or question_id in binding_complete_by_question
            ):
                raise HarnessError(
                    "complete binding records have invalid question IDs"
                )
            binding_complete_by_question[question_id] = binding
        if len(binding_complete_by_question) != 80:
            raise HarnessError(
                "complete binding records must cover 80 source problems"
            )
    missing = sorted(set(tasks) - set(underspecified))
    if missing:
        raise HarnessError(
            f"underspecified records miss actionable tasks: {missing[:5]}"
        )
    if len(complete) != 80:
        raise HarnessError(
            f"expected 80 complete controls, found {len(complete)}"
        )

    plan_tasks: dict[str, dict[str, Any]] = {}
    question_count = 0
    tasks_asking = 0
    for task_id, task in tasks.items():
        row = underspecified[task_id]
        if record_format == "protocol":
            if (
                row.get("prompt") != task["modified_content"]
                or row.get("id") != task["question_id"]
            ):
                raise HarnessError(f"{task_id}: protocol/data binding drift")
        else:
            if (
                row.get("question_id") != task["question_id"]
                or task_id not in binding_underspecified
                or binding_underspecified[task_id].get("id")
                != task["question_id"]
                or binding_underspecified[task_id].get("prompt")
                != task["modified_content"]
            ):
                raise HarnessError(f"{task_id}: ClarifyGPT/data binding drift")
        questions = questions_for(row, record_format=record_format)
        tasks_asking += bool(questions)
        question_count += len(questions)
        plan_tasks[task_id] = {
            "question_id": task["question_id"],
            "statement_sha256": sha256_text(str(task["modified_content"])),
            "questions": [
                {
                    "candidate_id": candidate_id(
                        method=method,
                        task_id=task_id,
                        position=position,
                        question=question,
                    ),
                    "text": question,
                    "queue_position": position,
                }
                for position, question in enumerate(questions)
            ],
        }

    control_rows: list[dict[str, Any]] = []
    for task_id in sorted(complete):
        row = complete[task_id]
        if record_format == "clarifygpt":
            binding = binding_complete_by_question.get(
                str(row.get("question_id"))
            )
            if (
                binding is None
                or binding.get("id") != row.get("question_id")
            ):
                raise HarnessError(
                    f"{task_id}: ClarifyGPT/control binding drift"
                )
        questions = questions_for(row, record_format=record_format)
        control_rows.append(
            {
                "task_id": task_id,
                "question_id": row.get("id") or row.get("question_id"),
                "asked": bool(questions),
                "question_count": len(questions),
            }
        )
    false_alarm_count = sum(row["asked"] for row in control_rows)

    hashes = {
        "dataset": sha256_bytes(dataset_path.read_bytes()),
        "underspecified_records": sha256_bytes(
            underspecified_path.read_bytes()
        ),
        "complete_records": sha256_bytes(complete_path.read_bytes()),
    }
    if binding_underspecified_path is not None:
        hashes["binding_underspecified_records"] = sha256_bytes(
            binding_underspecified_path.read_bytes()
        )
    if binding_complete_path is not None:
        hashes["binding_complete_records"] = sha256_bytes(
            binding_complete_path.read_bytes()
        )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "method": method,
        "benchmark_sha256": hashes["dataset"],
        "task_count": len(plan_tasks),
        "question_count": question_count,
        "tasks": plan_tasks,
        "selection": {
            "source": "frozen_native_baseline_question_trajectory",
            "record_format": record_format,
            "model": model,
        },
        "ordering": {
            "primary": "native_baseline_question_order",
            "downstream_stages_reorder": False,
            "gold_visible": False,
        },
        "input_hashes": hashes,
        "safety": {
            "new_baseline_generation_calls": 0,
            "gold_read_for_question_generation": False,
            "final_code_used": False,
        },
    }
    controls = {
        "schema_version": "ccb-v3-complete-control-outcomes-v1",
        "method": method,
        "model": model,
        "record_format": record_format,
        "complete_control_tasks": 80,
        "false_alarm_count": false_alarm_count,
        "F": false_alarm_count / 80,
        "tasks": control_rows,
        "input_hashes": hashes,
    }
    if tasks_asking != sum(
        bool(entry["questions"]) for entry in plan_tasks.values()
    ):
        raise HarnessError("actionable ask-count drift")
    return plan, controls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data_v3/ClarifyCodeBench-v3-actionable.jsonl"),
    )
    parser.add_argument("--underspecified", type=Path, required=True)
    parser.add_argument("--complete", type=Path, required=True)
    parser.add_argument(
        "--record-format",
        choices=("protocol", "clarifygpt"),
        required=True,
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--binding-underspecified", type=Path)
    parser.add_argument("--binding-complete", type=Path)
    parser.add_argument("--plan-out", type=Path, required=True)
    parser.add_argument("--controls-out", type=Path, required=True)
    args = parser.parse_args()
    plan, controls = build(
        dataset_path=args.data,
        underspecified_path=args.underspecified,
        complete_path=args.complete,
        record_format=args.record_format,
        method=args.method,
        model=args.model,
        binding_underspecified_path=args.binding_underspecified,
        binding_complete_path=args.binding_complete,
    )
    atomic_json(args.plan_out, plan)
    atomic_json(args.controls_out, controls)
    print(
        json.dumps(
            {
                "plan": str(args.plan_out),
                "plan_sha256": sha256_bytes(args.plan_out.read_bytes()),
                "actionable_tasks": plan["task_count"],
                "actionable_questions": plan["question_count"],
                "complete_controls": controls["complete_control_tasks"],
                "false_alarms": controls["false_alarm_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
