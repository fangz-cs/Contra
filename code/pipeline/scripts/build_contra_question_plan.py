#!/usr/bin/env python3
"""Build CONTRA's frozen ClarifyCodeBench queue from existing method authorities.

The main route is fixed to:

    Agent 1 decision-relevant
    -> Agent 2 V2 singleton unresolved
    -> self-generated BV with at least one stable behavioral difference

Downstream stages only remove candidates. Surviving questions retain their
native QG consolidation order (`node_position`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contra.question_plan import (
    PLAN_SCHEMA,
    HarnessError,
    index_tasks,
    read_jsonl,
    sha256_bytes,
    sha256_text,
)


def index_unique(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if (
            not isinstance(value, str)
            or not value
            or value in output
        ):
            raise HarnessError(f"{label}: invalid or duplicate {key}")
        output[value] = dict(row)
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


def build_plan(
    *,
    dataset_path: Path,
    candidate_input_path: Path,
    candidate_metadata_path: Path,
    v2_singleton_path: Path,
    self_bv_path: Path,
    method: str,
) -> dict[str, Any]:
    tasks = index_tasks(read_jsonl(dataset_path))
    candidates = index_unique(
        read_jsonl(candidate_input_path), "candidate_id", "candidate input"
    )
    metadata = index_unique(
        read_jsonl(candidate_metadata_path),
        "candidate_id",
        "candidate metadata",
    )
    v2 = index_unique(
        read_jsonl(v2_singleton_path), "candidate_id", "V2 singleton"
    )
    self_bv = index_unique(
        read_jsonl(self_bv_path), "candidate_id", "self BV"
    )
    if (
        not candidates
        or set(candidates) != set(metadata)
        or set(candidates) != set(v2)
        or not set(self_bv) <= set(candidates)
    ):
        raise HarnessError("authority coverage mismatch")

    agent1_gate = {
        candidate_id
        for candidate_id, row in v2.items()
        if row.get("decision_relevant") is True
    }
    if set(self_bv) != agent1_gate:
        raise HarnessError("self-BV coverage differs from Agent 1 gate")

    v2_gate = {
        candidate_id
        for candidate_id in self_bv
        if v2[candidate_id].get(
            "retained_for_behavioral_verification"
        ) is True
    }
    route = {
        candidate_id
        for candidate_id in v2_gate
        if int(self_bv[candidate_id].get("n_different") or 0) >= 1
    }
    controls = {
        candidate_id
        for candidate_id in route
        if candidate_id.startswith("s_task_")
    }
    by_task: dict[str, list[dict[str, Any]]] = {
        task_id: [] for task_id in tasks
    }
    position_census: dict[str, set[int]] = {
        task_id: set() for task_id in tasks
    }
    for candidate_id in sorted(route - controls):
        raw_task_id = candidate_id.split("::", 1)[0]
        if not raw_task_id.startswith("v_task_"):
            raise HarnessError(f"unexpected route candidate: {candidate_id}")
        task_id = raw_task_id[2:]
        if task_id not in tasks:
            raise HarnessError(f"route references unknown task: {task_id}")
        candidate = candidates[candidate_id]
        meta = metadata[candidate_id]
        question = candidate.get("clarification_question")
        requirement = candidate.get("complete_requirement")
        position = meta.get("node_position")
        if (
            not isinstance(question, str)
            or not question.strip()
            or requirement != tasks[task_id]["modified_content"]
            or meta.get("item_id") != raw_task_id
            or meta.get("node_validation_status")
            not in {"pass", "unresolved"}
            or not isinstance(position, int)
            or isinstance(position, bool)
            or position < 0
            or position in position_census[task_id]
        ):
            raise HarnessError(f"{candidate_id}: invalid QG binding")
        position_census[task_id].add(position)
        by_task[task_id].append(
            {
                "candidate_id": candidate_id,
                "text": question.strip(),
                "qg_position": position,
            }
        )

    plan_tasks: dict[str, dict[str, Any]] = {}
    selected_question_count = 0
    exact_duplicate_extras = 0
    task_question_distribution: Counter[int] = Counter()
    for task_id, task in tasks.items():
        selected = by_task[task_id]
        selected.sort(
            key=lambda row: (row["qg_position"], row["candidate_id"])
        )
        unique: list[dict[str, Any]] = []
        first_by_text: dict[str, dict[str, Any]] = {}
        for row in selected:
            first = first_by_text.get(row["text"])
            if first is None:
                row["duplicate_candidate_ids"] = []
                first_by_text[row["text"]] = row
                unique.append(row)
            else:
                first["duplicate_candidate_ids"].append(row["candidate_id"])
                exact_duplicate_extras += 1
        questions = [
            {
                "candidate_id": row["candidate_id"],
                "text": row["text"],
                "queue_position": queue_position,
                "duplicate_candidate_ids": row["duplicate_candidate_ids"],
            }
            for queue_position, row in enumerate(unique)
        ]
        selected_question_count += len(questions)
        task_question_distribution[len(questions)] += 1
        plan_tasks[task_id] = {
            "question_id": task["question_id"],
            "statement_sha256": sha256_text(task["modified_content"]),
            "questions": questions,
        }
    return {
        "schema_version": PLAN_SCHEMA,
        "method": method,
        "benchmark_sha256": sha256_bytes(dataset_path.read_bytes()),
        "task_count": len(plan_tasks),
        "question_count": selected_question_count,
        "tasks": plan_tasks,
        "selection": {
            "agent1": "decision_relevant",
            "agent2": "v2_singleton_unresolved",
            "behavioral_verification": (
                "self_generated_inputs_existential_difference_only"
            ),
        },
        "ordering": {
            "primary": "qg_consolidation_node_position",
            "final_tiebreak": "candidate_id",
            "downstream_stages_reorder": False,
            "exact_duplicate_policy": (
                "retain_earliest_qg_position_then_candidate_id"
            ),
            "gold_visible": False,
        },
        "counts": {
            "candidate_universe": len(candidates),
            "agent1_candidates": len(agent1_gate),
            "v2_gate_candidates": len(v2_gate),
            "route_candidates_all_cohorts": len(route),
            "route_control_candidates_excluded_from_plan": len(controls),
            "route_variant_candidates": len(route - controls),
            "exact_duplicate_candidate_questions_collapsed": (
                exact_duplicate_extras
            ),
            "route_variant_questions": selected_question_count,
            "tasks_with_questions": sum(bool(values) for values in by_task.values()),
            "tasks_without_questions": sum(
                not values for values in by_task.values()
            ),
            "task_question_distribution": {
                str(key): task_question_distribution[key]
                for key in sorted(task_question_distribution)
            },
        },
        "input_hashes": {
            "dataset": sha256_bytes(dataset_path.read_bytes()),
            "candidate_input": sha256_bytes(candidate_input_path.read_bytes()),
            "candidate_metadata": sha256_bytes(
                candidate_metadata_path.read_bytes()
            ),
            "v2_singleton": sha256_bytes(v2_singleton_path.read_bytes()),
            "self_bv": sha256_bytes(self_bv_path.read_bytes()),
        },
        "safety": {
            "gold_read": False,
            "expected_outputs_read": False,
            "oracle_inputs_used": False,
            "new_llm_calls": 0,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("inputs/clarifycodebench.jsonl"),
    )
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--v2-singleton", type=Path, required=True)
    parser.add_argument("--self-bv", type=Path, required=True)
    parser.add_argument("--method", default="CONTRA")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = build_plan(
        dataset_path=args.data,
        candidate_input_path=args.candidate_input,
        candidate_metadata_path=args.candidate_metadata,
        v2_singleton_path=args.v2_singleton,
        self_bv_path=args.self_bv,
        method=args.method,
    )
    atomic_json(args.out, plan)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                **plan["counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
