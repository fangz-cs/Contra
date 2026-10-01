"""Scoring helpers for question-only ClarifyCodeBench-v3 comparisons."""
from __future__ import annotations

from collections import defaultdict
from statistics import NormalDist
from typing import Any, Iterable, Mapping

from .ccb_v3_harness import HarnessError


def finished_by_task(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return the latest finished row for each task."""
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise HarnessError("run row has invalid task_id")
        if row.get("finished") is True:
            output[task_id] = dict(row)
    return output


def per_task(
    *,
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    official_metrics: Any,
    question_budget: int,
) -> dict[str, Any]:
    if record.get("prompt") != task.get("modified_content"):
        raise HarnessError(f"{task['task_id']}: run/data prompt mismatch")
    hit_dict = record.get("hit_dict")
    if not isinstance(hit_dict, dict):
        raise HarnessError(f"{task['task_id']}: missing hit_dict")
    indexed: list[tuple[int, int]] = []
    for raw_index, raw_value in hit_dict.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise HarnessError(
                f"{task['task_id']}: invalid hit index"
            ) from exc
        if (
            str(index) != str(raw_index)
            or not isinstance(raw_value, int)
            or isinstance(raw_value, bool)
            or raw_value not in {0, 1}
        ):
            raise HarnessError(f"{task['task_id']}: invalid hit value")
        if index < question_budget:
            indexed.append((index, raw_value))
    indexed.sort()
    if [index for index, _ in indexed] != list(range(len(indexed))):
        raise HarnessError(f"{task['task_id']}: non-contiguous hit sequence")

    hit_sequence = [value for _, value in indexed]
    question_turns = len(hit_sequence)
    key_questions = len(task.get("qa_pairs") or [])
    hits = sum(hit_sequence)
    if hits > key_questions:
        raise HarnessError(f"{task['task_id']}: hits exceed Gold")

    normalized_record = {
        "key_question_num": key_questions,
        "hit_dict": {
            str(index): value for index, value in enumerate(hit_sequence)
        },
        "turns": question_turns + 1,
    }
    official = official_metrics.score_record(normalized_record)
    return {
        "task_id": task["task_id"],
        "source": task["question_id"],
        "K": key_questions,
        "hits": hits,
        "question_turns": question_turns,
        "kqr": hits / key_questions if key_questions else None,
        "tkqr": float(official["tkqr"]),
        "ora": float(official["ora"]),
        "full_coverage": (
            float(hits == key_questions) if key_questions else None
        ),
        "asked_on_k0": float(key_questions == 0 and question_turns > 0),
        "correct_abstention": float(
            key_questions == 0 and question_turns == 0
        ),
    }


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise HarnessError("cannot summarize an empty run")
    positive = [row for row in rows if int(row["K"]) > 0]
    zero = [row for row in rows if int(row["K"]) == 0]
    total_gold = sum(int(row["K"]) for row in positive)
    total_hits = sum(int(row["hits"]) for row in positive)
    positive_count = len(positive)
    zero_count = len(zero)
    kqr_macro = (
        sum(float(row["kqr"]) for row in positive) / positive_count
        if positive_count
        else None
    )
    h_count = sum(int(row["question_turns"]) > 0 for row in positive)
    return {
        "tasks": len(rows),
        "actionable_tasks": positive_count,
        "zero_gold_tasks": zero_count,
        "gold": total_gold,
        "hits": total_hits,
        "KQR": kqr_macro,
        "KQR_micro": total_hits / total_gold if total_gold else None,
        "KQR_macro": kqr_macro,
        "TKQR": (
            sum(float(row["tkqr"]) for row in positive) / positive_count
            if positive_count
            else None
        ),
        "ORA": (
            sum(float(row["ora"]) for row in positive) / positive_count
            if positive_count
            else None
        ),
        "H_count": h_count,
        "H": h_count / positive_count if positive_count else None,
        "full_coverage": (
            sum(float(row["full_coverage"]) for row in positive)
            / positive_count
            if positive_count
            else None
        ),
        "avg_question_turns": (
            sum(int(row["question_turns"]) for row in positive)
            / positive_count
            if positive_count
            else None
        ),
        "TKQR_all_tasks_diagnostic": (
            sum(float(row["tkqr"]) for row in rows) / len(rows)
        ),
        "ORA_all_tasks_diagnostic": (
            sum(float(row["ora"]) for row in rows) / len(rows)
        ),
        "avg_question_turns_all_tasks_diagnostic": (
            sum(int(row["question_turns"]) for row in rows) / len(rows)
        ),
        "zero_gold_ask_rate": (
            sum(float(row["asked_on_k0"]) for row in zero) / zero_count
            if zero_count
            else None
        ),
        "zero_gold_correct_abstention": (
            sum(float(row["correct_abstention"]) for row in zero) / zero_count
            if zero_count
            else None
        ),
    }


def loglinear_d_prime(
    *,
    hit_count: int,
    hit_total: int,
    false_alarm_count: int,
    false_alarm_total: int,
) -> float:
    """Return log-linear-corrected d-prime for task-level ask decisions."""
    for label, count, total in (
        ("hit", hit_count, hit_total),
        ("false alarm", false_alarm_count, false_alarm_total),
    ):
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not isinstance(total, int)
            or isinstance(total, bool)
            or total <= 0
            or not 0 <= count <= total
        ):
            raise HarnessError(f"invalid {label} count")
    normal = NormalDist()
    hit_rate = (hit_count + 0.5) / (hit_total + 1)
    false_alarm_rate = (false_alarm_count + 0.5) / (
        false_alarm_total + 1
    )
    return normal.inv_cdf(hit_rate) - normal.inv_cdf(false_alarm_rate)


def group_by_source(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["source"])].append(dict(row))
    return dict(output)
