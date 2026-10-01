#!/usr/bin/env python3
"""Build Gold-blind question plans and source-control FPR for every gate."""
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

from forkcast.ccb_v3_harness import (
    PLAN_SCHEMA,
    HarnessError,
    index_tasks,
    read_jsonl,
    sha256_bytes,
    sha256_text,
)

STAGES = ("qg", "agent1", "agent2", "self_bv", "final")


def index_unique(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in output:
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


def render_plan(
    *,
    dataset_path: Path,
    tasks: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    route: set[str],
    method: str,
    stage: str,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
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
            or meta.get("node_validation_status") not in {"pass", "unresolved"}
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
    question_count = 0
    duplicate_count = 0
    distribution: Counter[int] = Counter()
    for task_id, task in tasks.items():
        selected = sorted(
            by_task[task_id],
            key=lambda row: (row["qg_position"], row["candidate_id"]),
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
                duplicate_count += 1
        questions = [
            {
                "candidate_id": row["candidate_id"],
                "text": row["text"],
                "queue_position": position,
                "duplicate_candidate_ids": row["duplicate_candidate_ids"],
            }
            for position, row in enumerate(unique)
        ]
        question_count += len(questions)
        distribution[len(questions)] += 1
        plan_tasks[task_id] = {
            "question_id": task["question_id"],
            "statement_sha256": sha256_text(str(task["modified_content"])),
            "questions": questions,
        }

    return {
        "schema_version": PLAN_SCHEMA,
        "method": method,
        "benchmark_sha256": sha256_bytes(dataset_path.read_bytes()),
        "task_count": len(tasks),
        "question_count": question_count,
        "tasks": plan_tasks,
        "selection": {
            "diagnostic_stage": stage,
            "stage_candidates_are_cumulative": True,
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
            "route_candidates_all_cohorts": len(route),
            "route_control_candidates_excluded_from_plan": len(controls),
            "route_variant_candidates": len(route - controls),
            "exact_duplicate_candidate_questions_collapsed": duplicate_count,
            "route_variant_questions": question_count,
            "tasks_with_questions": sum(bool(rows) for rows in by_task.values()),
            "tasks_without_questions": sum(
                not rows for rows in by_task.values()
            ),
            "task_question_distribution": {
                str(key): distribution[key] for key in sorted(distribution)
            },
        },
        "input_hashes": dict(input_hashes),
        "safety": {
            "gold_read_for_selection": False,
            "expected_outputs_read": False,
            "oracle_inputs_used": False,
            "new_method_llm_calls": 0,
        },
    }


def build_all(
    *,
    dataset_path: Path,
    snapshot_path: Path,
    candidate_input_path: Path,
    candidate_metadata_path: Path,
    agent1_path: Path,
    agent2_path: Path,
    self_bv_path: Path,
    method_label: str,
    output_dir: Path,
    final_output_path: Path,
    stages: tuple[str, ...] = STAGES,
    report_output_path: Path | None = None,
) -> dict[str, Any]:
    if not stages or stages != STAGES[: len(stages)]:
        raise HarnessError("stages must be a non-empty cumulative prefix")
    tasks = index_tasks(read_jsonl(dataset_path))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_rows = snapshot.get("tasks") if isinstance(snapshot, dict) else None
    if not isinstance(snapshot_rows, list):
        raise HarnessError("selection snapshot has no task list")
    snapshot_ids = {
        str(row.get("item_id"))
        for row in snapshot_rows
        if isinstance(row, Mapping) and row.get("item_id")
    }
    snapshot_variants = {
        item_id for item_id in snapshot_ids if item_id.startswith("v_task_")
    }
    source_tasks = {
        item_id for item_id in snapshot_ids if item_id.startswith("s_task_")
    }
    candidates = index_unique(
        read_jsonl(candidate_input_path), "candidate_id", "candidate input"
    )
    metadata = index_unique(
        read_jsonl(candidate_metadata_path), "candidate_id", "metadata"
    )
    universe = set(candidates)
    if not universe or set(metadata) != universe:
        raise HarnessError("stage authority coverage mismatch")

    routes = {"qg": universe}
    agent1_ids: set[str] = set()
    agent2_ids: set[str] = set()
    bv_ids: set[str] = set()
    if "agent1" in stages:
        agent1 = index_unique(
            read_jsonl(agent1_path), "candidate_id", "Agent1"
        )
        if set(agent1) != universe:
            raise HarnessError("stage authority coverage mismatch")
        agent1_ids = {
            candidate_id
            for candidate_id, row in agent1.items()
            if row.get("decision_relevant") is True
        }
        routes["agent1"] = agent1_ids
    if "agent2" in stages:
        agent2 = index_unique(
            read_jsonl(agent2_path), "candidate_id", "Agent2"
        )
        if set(agent2) != universe:
            raise HarnessError("stage authority coverage mismatch")
        agent2_ids = {
            candidate_id
            for candidate_id, row in agent2.items()
            if row.get("retained_for_behavioral_verification") is True
        }
        if not agent2_ids <= agent1_ids:
            raise HarnessError("Agent2 gate is not a subset of Agent1")
        routes["agent2"] = agent2_ids
    if "self_bv" in stages:
        self_bv = index_unique(
            read_jsonl(self_bv_path), "candidate_id", "self-BV"
        )
        if set(self_bv) != agent1_ids:
            raise HarnessError("self-BV coverage differs from Agent1 gate")
        bv_ids = {
            candidate_id
            for candidate_id, row in self_bv.items()
            if isinstance(row.get("n_different"), int)
            and not isinstance(row.get("n_different"), bool)
            and row["n_different"] >= 1
        }
        if not bv_ids <= agent1_ids:
            raise HarnessError("self-BV gate is not a subset of Agent1")
        routes["self_bv"] = bv_ids
    if "final" in stages:
        routes["final"] = agent2_ids & bv_ids

    metadata_items = {str(row["item_id"]) for row in metadata.values()}
    actionable_tasks = sum(bool(task.get("qa_pairs")) for task in tasks.values())
    gold_questions = sum(
        len(task.get("qa_pairs") or []) for task in tasks.values()
    )
    zero_gold_tasks = len(tasks) - actionable_tasks
    if (
        len(tasks) != 419
        or actionable_tasks != 328
        or gold_questions != 463
        or zero_gold_tasks != 91
        or len(source_tasks) != 80
        or len(snapshot_variants) != 328
        or len(snapshot_ids) != 408
        or not metadata_items <= snapshot_ids
    ):
        raise HarnessError("unexpected evaluation denominator census")

    input_hashes: dict[str, str] = {
        "dataset": sha256_bytes(dataset_path.read_bytes()),
        "selection_snapshot": sha256_bytes(snapshot_path.read_bytes()),
        "candidate_input": sha256_bytes(candidate_input_path.read_bytes()),
        "candidate_metadata": sha256_bytes(
            candidate_metadata_path.read_bytes()
        ),
    }
    if "agent1" in stages:
        input_hashes["agent1"] = sha256_bytes(agent1_path.read_bytes())
    if "agent2" in stages:
        input_hashes["v2_singleton"] = sha256_bytes(
            agent2_path.read_bytes()
        )
    if "self_bv" in stages:
        input_hashes["self_bv"] = sha256_bytes(self_bv_path.read_bytes())
    output_dir.mkdir(parents=True, exist_ok=True)
    report_stages: dict[str, dict[str, Any]] = {}
    for stage in stages:
        label = (
            method_label
            if stage == "final"
            else f"{method_label}-Stage-{stage.upper()}"
        )
        plan_path = (
            final_output_path
            if stage == "final"
            else output_dir / f"stage-{stage}.json"
        )
        plan = render_plan(
            dataset_path=dataset_path,
            tasks=tasks,
            candidates=candidates,
            metadata=metadata,
            route=routes[stage],
            method=label,
            stage=stage,
            input_hashes={
                key: value
                for key, value in input_hashes.items()
                if (
                    key
                    in {
                        "dataset",
                        "selection_snapshot",
                        "candidate_input",
                        "candidate_metadata",
                    }
                    or key == "agent1"
                    and stage != "qg"
                    or key == "v2_singleton"
                    and stage in {"agent2", "final"}
                    or key == "self_bv"
                    and stage in {"self_bv", "final"}
                )
            },
        )
        if stage == "final":
            plan["counts"].update(
                {
                    "candidate_universe": len(universe),
                    "agent1_candidates": len(agent1_ids),
                    "v2_gate_candidates": len(agent2_ids),
                    "route_control_candidates_excluded_from_v3_plan": (
                        plan["counts"][
                            "route_control_candidates_excluded_from_plan"
                        ]
                    ),
                }
            )
            plan["selection"] = {
                "agent1": "decision_relevant",
                "agent2": "v2_singleton_unresolved",
                "behavioral_verification": (
                    "self_generated_inputs_existential_difference_only"
                ),
            }
        atomic_json(plan_path, plan)
        retained_source_tasks = {
            str(metadata[candidate_id]["item_id"])
            for candidate_id in routes[stage]
            if metadata[candidate_id].get("kind") == "source"
        }
        report_stages[stage] = {
            "method_label": label,
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": sha256_bytes(plan_path.read_bytes()),
            "candidate_count": len(routes[stage]),
            "variant_candidate_count": sum(
                metadata[candidate_id].get("kind") == "variant"
                for candidate_id in routes[stage]
            ),
            "source_control_candidate_count": sum(
                metadata[candidate_id].get("kind") == "source"
                for candidate_id in routes[stage]
            ),
            "source_control_tasks_retained": len(retained_source_tasks),
            "source_control_task_fpr": (
                len(retained_source_tasks) / len(source_tasks)
            ),
            "question_count_after_exact_deduplication": plan[
                "question_count"
            ],
        }

    report = {
        "schema_version": "ccb-v3-stage-question-plans-v1",
        "method_label": method_label,
        "recall_definition": (
            "official GPT-4o matcher hits within the first five questions "
            "divided by all 463 Gold questions"
        ),
        "fpr_definition": (
            "complete-source control tasks retaining at least one candidate "
            "divided by all 80 complete-source control tasks"
        ),
        "denominators": {
            "gold_questions": gold_questions,
            "actionable_variant_tasks": actionable_tasks,
            "complete_source_control_tasks": len(source_tasks),
            "harness_zero_gold_tasks": zero_gold_tasks,
        },
        "stages": report_stages,
        "input_hashes": input_hashes,
    }
    atomic_json(
        report_output_path or output_dir / "stage-plan-report.json",
        report,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--agent1", type=Path, required=True)
    parser.add_argument("--agent2", type=Path, required=True)
    parser.add_argument("--self-bv", type=Path, required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help="cumulative stage prefix, comma-separated",
    )
    parser.add_argument("--report-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_all(
        dataset_path=args.data,
        snapshot_path=args.snapshot,
        candidate_input_path=args.candidate_input,
        candidate_metadata_path=args.candidate_metadata,
        agent1_path=args.agent1,
        agent2_path=args.agent2,
        self_bv_path=args.self_bv,
        method_label=args.method_label,
        output_dir=args.output_dir,
        final_output_path=args.final_output,
        stages=tuple(args.stages.split(",")),
        report_output_path=args.report_output,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
