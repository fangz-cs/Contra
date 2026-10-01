#!/usr/bin/env python3
"""Rescore frozen QG/AV artifacts on ClarifyCodeBench-v3-actionable."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import audit_agent2_resolution_spans as util


SYSTEM_PROMPT = """\
You are evaluating clarification-question recall against a frozen benchmark.

For each generated candidate question, return every Gold question that asks the
same underlying clarification decision with the same scope.

Match semantically, not lexically. A candidate and a Gold question may use
different wording. However, topical relatedness is insufficient: do not match
questions that ask different boundaries, different operation stages, different
quantifiers, or different output requirements. A broader or bundled candidate
matches a Gold question only if answering that Gold question would fully
resolve the candidate's decision; partial overlap does not count.

Use the task specification and Gold answers only to disambiguate intent. Do not
judge whether a Gold question is useful, already resolved, or behaviorally
important: that filtering was performed when V3-actionable was constructed.
Evaluate every candidate independently and return strict JSON."""

SCHEMA_MANIFEST = "ccb-v3-frozen-av-evaluation-manifest-v1"
SCHEMA_CALL = "ccb-v3-frozen-av-evaluation-call-v1"
SCHEMA_MATCH = "ccb-v3-frozen-av-candidate-match-v1"
TRANSIENT_HTTP_STATUSES = frozenset(
    {408, 409, 425, 429, 499, 500, 502, 503, 504, 520, 521, 522, 524, 529}
)


class EvaluationError(RuntimeError):
    pass


def index_unique(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in output:
            raise EvaluationError(f"{label}: invalid or duplicate {key}")
        output[value] = dict(row)
    return output


def gate(path: Path) -> dict[str, bool]:
    output: dict[str, bool] = {}
    for row in util.read_jsonl(path):
        candidate_id = row.get("candidate_id")
        retained = row.get("retained_for_behavioral_verification")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in output
            or not isinstance(retained, bool)
        ):
            raise EvaluationError(f"{path}: invalid gate row")
        output[candidate_id] = retained
    return output


def sha256_file(path: Path) -> str:
    return util.sha256_file(path)


def stable_order(values: Sequence[str], salt: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{salt}\0{value}".encode("utf-8")
        ).hexdigest(),
    )


def load_units(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    candidates = index_unique(
        util.read_jsonl(args.candidate_input),
        "candidate_id",
        "candidate input",
    )
    metadata = index_unique(
        util.read_jsonl(args.candidate_metadata),
        "candidate_id",
        "candidate metadata",
    )
    if set(candidates) != set(metadata) or len(candidates) != 17877:
        raise EvaluationError("candidate/metadata census mismatch")
    v3_rows = util.read_jsonl(args.v3_actionable)
    v1 = index_unique(
        util.read_jsonl(args.v1_variants), "task_id", "V1 variants"
    )
    if len(v3_rows) != 328 or len(v1) != 419:
        raise EvaluationError("benchmark task census mismatch")
    units: list[dict[str, Any]] = []
    seen_gold: set[str] = set()
    v1_question_overlap = 0
    v1_all_exact_tasks = 0
    provenance_counts: Counter[str] = Counter()
    v3_candidate_ids: set[str] = set()
    for row in v3_rows:
        task_id = str(row.get("task_id"))
        item_id = "v_" + task_id
        gold_rows = row.get("qa_pairs")
        if (
            row.get("schema_version")
            != "ClarifyCodeBench-v3-actionable"
            or not isinstance(gold_rows, list)
            or len(gold_rows) != row.get("k")
        ):
            raise EvaluationError(f"{task_id}: invalid V3 row")
        task_candidates = sorted(
            candidate_id
            for candidate_id in candidates
            if candidate_id.startswith(item_id + "::")
        )
        if not task_candidates:
            raise EvaluationError(f"{task_id}: no frozen candidates")
        requirements = {
            str(candidates[candidate_id]["complete_requirement"])
            for candidate_id in task_candidates
        }
        question_ids = {
            str(metadata[candidate_id]["question_id"])
            for candidate_id in task_candidates
        }
        if requirements != {str(row["modified_content"])}:
            raise EvaluationError(f"{task_id}: requirement drift")
        if question_ids != {str(row["question_id"])}:
            raise EvaluationError(f"{task_id}: question_id drift")
        v3_candidate_ids.update(task_candidates)
        old_questions = {
            str(value["question"])
            for value in v1[task_id]["qa_pairs"]
        }
        gold: list[dict[str, Any]] = []
        exact_on_task = 0
        for index, value in enumerate(gold_rows, 1):
            gold_id = f"{item_id}::gold_{index:02d}"
            if gold_id in seen_gold:
                raise EvaluationError(f"{gold_id}: duplicate Gold ID")
            seen_gold.add(gold_id)
            question = str(value["question"])
            exact = question in old_questions
            exact_on_task += int(exact)
            v1_question_overlap += int(exact)
            provenance = str(value.get("provenance"))
            provenance_counts[provenance] += 1
            gold.append(
                {
                    "gold_id": gold_id,
                    "question": question,
                    "answer": str(value["answer"]),
                    "provenance": provenance,
                    "exact_v1_question": exact,
                }
            )
        v1_all_exact_tasks += int(exact_on_task == len(gold))
        units.append(
            {
                "schema_version": "ccb-v3-frozen-evaluation-unit-v1",
                "item_id": item_id,
                "task_id": task_id,
                "question_id": str(row["question_id"]),
                "requirement": str(row["modified_content"]),
                "gold": gold,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "question": str(
                            candidates[candidate_id][
                                "clarification_question"
                            ]
                        ),
                    }
                    for candidate_id in task_candidates
                ],
            }
        )
    if len(seen_gold) != 463:
        raise EvaluationError("V3 Gold census mismatch")
    overlap = {
        "v3_tasks": len(units),
        "v3_gold": len(seen_gold),
        "v3_candidate_count": len(v3_candidate_ids),
        "requirements_exact": len(units),
        "exact_v1_question_overlap": v1_question_overlap,
        "all_questions_exact_v1_tasks": v1_all_exact_tasks,
        "provenance_counts": dict(sorted(provenance_counts.items())),
    }
    return units, candidates, metadata, overlap


def build_batches(
    units: Sequence[Mapping[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for unit in units:
        ordered = sorted(
            unit["candidates"],
            key=lambda row: hashlib.sha256(
                str(row["candidate_id"]).encode("utf-8")
            ).hexdigest(),
        )
        for index in range(0, len(ordered), batch_size):
            batches.append(
                {
                    **unit,
                    "batch_id": (
                        f"{unit['item_id']}::batch_"
                        f"{index // batch_size:03d}"
                    ),
                    "candidates": ordered[index : index + batch_size],
                }
            )
    return batches


def public_unit(
    unit: Mapping[str, Any], replicate: int
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    candidate_by_id = {
        str(row["candidate_id"]): row for row in unit["candidates"]
    }
    gold_by_id = {str(row["gold_id"]): row for row in unit["gold"]}
    candidate_ids = stable_order(
        list(candidate_by_id), f"candidate:{replicate}:{unit['item_id']}"
    )
    gold_ids = stable_order(
        list(gold_by_id), f"gold:{replicate}:{unit['item_id']}"
    )
    local_candidate_for = {
        candidate_id: f"c{index:03d}"
        for index, candidate_id in enumerate(candidate_ids, 1)
    }
    local_gold_for = {
        gold_id: f"g{index:02d}"
        for index, gold_id in enumerate(gold_ids, 1)
    }
    payload = {
        "task_specification": unit["requirement"],
        "gold_questions": [
            {
                "gold_id": local_gold_for[gold_id],
                "question": gold_by_id[gold_id]["question"],
                "answer": gold_by_id[gold_id]["answer"],
            }
            for gold_id in gold_ids
        ],
        "candidate_questions": [
            {
                "candidate_id": local_candidate_for[candidate_id],
                "question": candidate_by_id[candidate_id]["question"],
            }
            for candidate_id in candidate_ids
        ],
    }
    return (
        payload,
        {local: original for original, local in local_candidate_for.items()},
        {local: original for original, local in local_gold_for.items()},
    )


def response_schema(
    candidate_ids: Sequence[str], gold_ids: Sequence[str]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidate_matches": {
                "type": "array",
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {
                            "type": "string",
                            "enum": list(candidate_ids),
                        },
                        "gold_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(gold_ids),
                            },
                        },
                    },
                    "required": ["candidate_id", "gold_ids"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidate_matches"],
        "additionalProperties": False,
    }


def parse_response(
    text: str,
    *,
    candidate_ids: Sequence[str],
    gold_ids: Sequence[str],
) -> list[dict[str, Any]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvaluationError("response is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"candidate_matches"}:
        raise EvaluationError("response root is invalid")
    rows = value["candidate_matches"]
    if not isinstance(rows, list):
        raise EvaluationError("candidate_matches is not a list")
    expected_candidates = set(candidate_ids)
    expected_gold = set(gold_ids)
    parsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "candidate_id",
            "gold_ids",
        }:
            raise EvaluationError("candidate match fields are invalid")
        candidate_id = row["candidate_id"]
        matches = row["gold_ids"]
        if (
            not isinstance(candidate_id, str)
            or candidate_id not in expected_candidates
            or candidate_id in parsed
            or not isinstance(matches, list)
            or len(matches) != len(set(matches))
            or any(
                not isinstance(gold_id, str)
                or gold_id not in expected_gold
                for gold_id in matches
            )
        ):
            raise EvaluationError("candidate match value is invalid")
        parsed[candidate_id] = {
            "candidate_id": candidate_id,
            "gold_ids": sorted(matches),
        }
    if set(parsed) != expected_candidates:
        raise EvaluationError("candidate coverage mismatch")
    return [parsed[candidate_id] for candidate_id in candidate_ids]


def call_unit(
    *,
    unit: Mapping[str, Any],
    replicate: int,
    client: Any,
    model: str,
) -> dict[str, Any]:
    payload, candidate_original, gold_original = public_unit(
        unit, replicate
    )
    candidate_ids = [
        str(row["candidate_id"]) for row in payload["candidate_questions"]
    ]
    gold_ids = [str(row["gold_id"]) for row in payload["gold_questions"]]
    schema = response_schema(candidate_ids, gold_ids)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]
    retry_reasons: list[dict[str, Any]] = []
    protocol_errors: list[str] = []
    started = time.monotonic()
    for protocol_attempt in range(1, 4):
        response = None
        for transport_attempt in range(1, 13):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=16384,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ccb_v3_candidate_gold_matching",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                retry_reasons.append(
                    {
                        "attempt": transport_attempt,
                        "status_code": status,
                        "error": f"{type(exc).__name__}: {exc}"[:4000],
                    }
                )
                permanent = (
                    isinstance(status, int)
                    and 400 <= status < 500
                    and status not in TRANSIENT_HTTP_STATUSES
                )
                if permanent or transport_attempt >= 12:
                    raise
                if status == 429:
                    delay = min(120.0, 15.0 * transport_attempt)
                    delay += random.uniform(0.0, 5.0)
                else:
                    delay = min(
                        30.0, 0.75 * 2 ** (transport_attempt - 1)
                    )
                time.sleep(delay)
        if response is None or not response.choices:
            raise EvaluationError("provider returned no choices")
        text = getattr(response.choices[0].message, "content", None) or ""
        try:
            local_matches = parse_response(
                text,
                candidate_ids=candidate_ids,
                gold_ids=gold_ids,
            )
        except EvaluationError as exc:
            protocol_errors.append(str(exc))
            if protocol_attempt >= 3:
                raise
            messages.extend(
                [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            f"Protocol error: {exc}. Return corrected JSON "
                            "covering every supplied candidate exactly once."
                        ),
                    },
                ]
            )
            continue
        matches = [
            {
                "candidate_id": candidate_original[row["candidate_id"]],
                "gold_ids": sorted(
                    gold_original[gold_id]
                    for gold_id in row["gold_ids"]
                ),
            }
            for row in local_matches
        ]
        usage = getattr(response, "usage", None)
        return {
            "schema_version": SCHEMA_CALL,
            "call_key": f"{unit['batch_id']}::replicate_{replicate}",
            "item_id": unit["item_id"],
            "batch_id": unit["batch_id"],
            "question_id": unit["question_id"],
            "replicate": replicate,
            "input_sha256": util.stable_json_sha256(payload),
            "model": model,
            "temperature": 0.0,
            "matches": matches,
            "retry_reasons": retry_reasons,
            "protocol_errors": protocol_errors,
            "usage": (
                usage.model_dump()
                if hasattr(usage, "model_dump")
                else {}
            ),
            "latency_seconds": round(time.monotonic() - started, 3),
            "completed_at": util.utc_now(),
        }
    raise AssertionError("unreachable")


def bootstrap_delta(
    *,
    gold_rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        by_cluster[str(row["question_id"])].append(row)
    clusters = sorted(by_cluster)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        selected = rng.choices(clusters, k=len(clusters))
        numerator = 0
        denominator = 0
        for cluster in selected:
            rows = by_cluster[cluster]
            numerator += sum(
                int(row["stage_hits"][right])
                - int(row["stage_hits"][left])
                for row in rows
            )
            denominator += len(rows)
        values.append(numerator / denominator)
    values.sort()
    point = (
        sum(
            int(row["stage_hits"][right])
            - int(row["stage_hits"][left])
            for row in gold_rows
        )
        / len(gold_rows)
    )
    return {
        "left": left,
        "right": right,
        "point_delta": point,
        "cluster_count": len(clusters),
        "bootstrap_samples": samples,
        "seed": seed,
        "ci95": [
            values[int(0.025 * samples)],
            values[int(0.975 * samples) - 1],
        ],
    }


def fleiss_kappa(pair_votes: Sequence[int], replicates: int) -> float:
    if not pair_votes:
        raise EvaluationError("pair-vote census is empty")
    observed = sum(
        (
            votes * (votes - 1)
            + (replicates - votes) * (replicates - votes - 1)
        )
        / (replicates * (replicates - 1))
        for votes in pair_votes
    ) / len(pair_votes)
    yes = sum(pair_votes) / (len(pair_votes) * replicates)
    expected = yes * yes + (1 - yes) * (1 - yes)
    return (observed - expected) / (1 - expected)


def render_report(output: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen AV evaluation on ClarifyCodeBench-v3-actionable",
        "",
        "No QG, Agent 1, Agent 2, program generation, or BV call was rerun.",
        "",
        "## Alignment",
        "",
        f"- Exact requirements: {output['overlap']['requirements_exact']}/328",
        (
            "- Exact V1 Gold-question overlap: "
            f"{output['overlap']['exact_v1_question_overlap']}/463"
        ),
        "",
        "## Gold recall",
        "",
        "| Stage | Gold recall | All-Gold task coverage | Candidates |",
        "|---|---:|---:|---:|",
    ]
    for stage in output["stage_order"]:
        row = output["stages"][stage]
        lines.append(
            f"| {stage} | {row['gold_hits']}/{row['gold_total']} "
            f"({100 * row['gold_recall']:.2f}%) "
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
    lines.extend(
        [
            "",
            "## Evaluator agreement",
            "",
            (
                f"- Fleiss' kappa across candidate-Gold pairs: "
                f"{output['evaluator_agreement']['fleiss_kappa']:.4f}"
            ),
            (
                f"- Unanimous pairs: "
                f"{output['evaluator_agreement']['unanimous_pairs']}/"
                f"{output['evaluator_agreement']['pair_count']}"
            ),
            "",
            (
                "V3 is a provisional correction snapshot. Gold metrics here "
                "are development evidence, not a replacement for a fully "
                "independently validated benchmark authority."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def finalize(
    *,
    args: argparse.Namespace,
    units: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    overlap: Mapping[str, Any],
    calls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    stage_gates = {
        "QG": {candidate_id: True for candidate_id in candidates},
        "Agent 1": gate(args.agent1_gate),
        "Agent 1 + Agent 2 v1": gate(args.agent2_v1_gate),
        "Agent 1 + Agent 2 v2": gate(args.agent2_v2_gate),
    }
    if any(set(values) != set(candidates) for values in stage_gates.values()):
        raise EvaluationError("gate candidate coverage mismatch")
    if not all(
        (
            not stage_gates["Agent 1 + Agent 2 v1"][candidate_id]
            or stage_gates["Agent 1"][candidate_id]
        )
        and (
            not stage_gates["Agent 1 + Agent 2 v2"][candidate_id]
            or stage_gates["Agent 1"][candidate_id]
        )
        for candidate_id in candidates
    ):
        raise EvaluationError("Agent 2 gate is not an Agent 1 subset")

    vote_counts: Counter[tuple[str, str]] = Counter()
    pair_universe: set[tuple[str, str]] = set()
    for unit in units:
        candidate_ids = {
            str(row["candidate_id"]) for row in unit["candidates"]
        }
        gold_ids = {str(row["gold_id"]) for row in unit["gold"]}
        pair_universe.update(
            (candidate_id, gold_id)
            for candidate_id in candidate_ids
            for gold_id in gold_ids
        )
    for call in calls.values():
        for row in call["matches"]:
            for gold_id in row["gold_ids"]:
                vote_counts[(row["candidate_id"], gold_id)] += 1
    match_rows: list[dict[str, Any]] = []
    matches_by_gold: dict[str, set[str]] = defaultdict(set)
    for candidate_id, gold_id in sorted(pair_universe):
        votes = vote_counts[(candidate_id, gold_id)]
        if votes >= args.replicates // 2 + 1:
            matches_by_gold[gold_id].add(candidate_id)
        match_rows.append(
            {
                "schema_version": SCHEMA_MATCH,
                "candidate_id": candidate_id,
                "gold_id": gold_id,
                "votes": votes,
                "matched": votes >= args.replicates // 2 + 1,
            }
        )
    util.atomic_bytes(
        args.output_dir / "candidate_gold_pair_votes.jsonl",
        util.jsonl_bytes(match_rows),
    )

    stage_order = list(stage_gates)
    gold_results: list[dict[str, Any]] = []
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        for gold in unit["gold"]:
            gold_id = str(gold["gold_id"])
            matched = matches_by_gold[gold_id]
            stage_hits = {
                stage: any(values[candidate_id] for candidate_id in matched)
                for stage, values in stage_gates.items()
            }
            unanimous = {
                stage: any(
                    stage_gates[stage][candidate_id]
                    and vote_counts[(candidate_id, gold_id)]
                    == args.replicates
                    for candidate_id in matched
                )
                for stage in stage_order
            }
            any_vote = {
                stage: any(
                    stage_gates[stage][candidate_id]
                    and vote_counts[(candidate_id, gold_id)] >= 1
                    for candidate_id, candidate_gold in pair_universe
                    if candidate_gold == gold_id
                )
                for stage in stage_order
            }
            row = {
                "gold_id": gold_id,
                "item_id": unit["item_id"],
                "question_id": unit["question_id"],
                "question": gold["question"],
                "answer": gold["answer"],
                "provenance": gold["provenance"],
                "exact_v1_question": gold["exact_v1_question"],
                "matched_candidate_ids": sorted(matched),
                "stage_hits": stage_hits,
                "unanimous_pair_stage_hits": unanimous,
                "any_vote_stage_hits": any_vote,
            }
            gold_results.append(row)
            tasks[str(unit["item_id"])].append(row)
    util.atomic_bytes(
        args.output_dir / "gold_results.jsonl",
        util.jsonl_bytes(gold_results),
    )

    v3_candidate_ids = {
        str(candidate["candidate_id"])
        for unit in units
        for candidate in unit["candidates"]
    }
    stages: dict[str, Any] = {}
    for stage in stage_order:
        hits = sum(row["stage_hits"][stage] for row in gold_results)
        all_gold = sum(
            all(row["stage_hits"][stage] for row in rows)
            for rows in tasks.values()
        )
        any_gold = sum(
            any(row["stage_hits"][stage] for row in rows)
            for rows in tasks.values()
        )
        stages[stage] = {
            "candidate_count": sum(
                stage_gates[stage][candidate_id]
                for candidate_id in v3_candidate_ids
            ),
            "gold_hits": hits,
            "gold_total": len(gold_results),
            "gold_recall": hits / len(gold_results),
            "all_gold_tasks": all_gold,
            "any_gold_tasks": any_gold,
            "task_total": len(tasks),
            "all_gold_task_rate": all_gold / len(tasks),
            "any_gold_task_rate": any_gold / len(tasks),
            "unanimous_pair_gold_hits": sum(
                row["unanimous_pair_stage_hits"][stage]
                for row in gold_results
            ),
            "any_vote_gold_hits": sum(
                row["any_vote_stage_hits"][stage]
                for row in gold_results
            ),
        }
    comparisons = [
        ("QG_to_Agent1", "QG", "Agent 1"),
        (
            "Agent1_to_Agent2_v1",
            "Agent 1",
            "Agent 1 + Agent 2 v1",
        ),
        (
            "Agent1_to_Agent2_v2",
            "Agent 1",
            "Agent 1 + Agent 2 v2",
        ),
        (
            "Agent2_v1_to_v2",
            "Agent 1 + Agent 2 v1",
            "Agent 1 + Agent 2 v2",
        ),
    ]
    paired_deltas = {
        name: bootstrap_delta(
            gold_rows=gold_results,
            left=left,
            right=right,
            samples=args.bootstrap_samples,
            seed=args.seed + index,
        )
        for index, (name, left, right) in enumerate(comparisons)
    }
    pair_votes = [
        vote_counts[pair] for pair in sorted(pair_universe)
    ]
    output = {
        "schema_version": "ccb-v3-frozen-av-evaluation-summary-v1",
        "completed_at": util.utc_now(),
        "benchmark_status": (
            "provisional correction snapshot; development evidence only"
        ),
        "no_generation_or_filtering_rerun": True,
        "overlap": dict(overlap),
        "stage_order": stage_order,
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
                votes >= args.replicates // 2 + 1
                for votes in pair_votes
            ),
            "fleiss_kappa": fleiss_kappa(
                pair_votes, args.replicates
            ),
        },
        "hashes": {
            "v3_actionable": sha256_file(args.v3_actionable),
            "v1_variants": sha256_file(args.v1_variants),
            "v1_v3_mapping": sha256_file(args.v1_v3_mapping),
            "candidate_input": sha256_file(args.candidate_input),
            "candidate_metadata": sha256_file(args.candidate_metadata),
            "agent1_gate": sha256_file(args.agent1_gate),
            "agent2_v1_gate": sha256_file(args.agent2_v1_gate),
            "agent2_v2_gate": sha256_file(args.agent2_v2_gate),
            "calls": sha256_file(args.output_dir / "calls.jsonl"),
            "pair_votes": sha256_file(
                args.output_dir / "candidate_gold_pair_votes.jsonl"
            ),
            "gold_results": sha256_file(
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
    mapping = json.loads(
        args.v1_v3_mapping.read_text(encoding="utf-8")
    )
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
        raise EvaluationError(
            "V1/V3 mapping requires a separate generation run"
        )
    units, candidates, metadata, overlap = load_units(args)
    batches = build_batches(units, args.candidate_batch_size)
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
        "task_count": len(units),
        "gold_count": sum(len(unit["gold"]) for unit in units),
        "candidate_count": sum(
            len(unit["candidates"]) for unit in units
        ),
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": util.stable_json_sha256(SYSTEM_PROMPT),
        "gate_visibility_to_evaluator": False,
        "v3_actionable_sha256": sha256_file(args.v3_actionable),
        "v1_variants_sha256": sha256_file(args.v1_variants),
        "v1_v3_mapping_sha256": sha256_file(args.v1_v3_mapping),
        "candidate_input_sha256": sha256_file(args.candidate_input),
        "candidate_metadata_sha256": sha256_file(
            args.candidate_metadata
        ),
        "units_sha256": sha256_file(units_path),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "overlap": overlap,
    }
    manifest_path = args.output_dir / "manifest.json"
    util.write_immutable(manifest_path, util.json_bytes(manifest))
    if args.prepare_only:
        return None
    if not args.confirm_live:
        raise EvaluationError("live evaluation requires --confirm-live")
    complete_path = args.output_dir / "COMPLETE.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            complete.get("manifest_sha256") != sha256_file(manifest_path)
            or complete.get("summary_sha256")
            != sha256_file(args.output_dir / "summary.json")
        ):
            raise EvaluationError("existing completion failed integrity")
        return json.loads(
            (args.output_dir / "summary.json").read_text(encoding="utf-8")
        )
    base_url = os.environ.get("CONTRA_BASE_URL")
    api_key = os.environ.get("CONTRA_API_KEY")
    if not base_url or not api_key:
        raise EvaluationError("set CONTRA_BASE_URL and CONTRA_API_KEY")
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
    existing = index_unique(
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
                call_unit,
                unit=unit,
                replicate=replicate,
                client=client,
                model=args.model,
            ): (str(unit["item_id"]), replicate)
            for unit, replicate in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            item_id, replicate = futures[future]
            try:
                row = future.result()
                util.append_jsonl(calls_path, row, lock=lock)
                existing[row["call_key"]] = row
            except Exception as exc:
                failures.append(
                    f"{item_id}/r{replicate}: "
                    f"{type(exc).__name__}: {exc}"
                )
            if index % 20 == 0 or index == len(pending):
                print(
                    f"{util.utc_now()} calls={len(existing)}/"
                    f"{len(tasks)} failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise EvaluationError(
            f"{len(failures)} calls failed: {' | '.join(failures[:8])}"
        )
    if set(existing) != {
        f"{batch['batch_id']}::replicate_{replicate}"
        for batch, replicate in tasks
    }:
        raise EvaluationError("call coverage mismatch")
    output = finalize(
        args=args,
        units=units,
        candidates=candidates,
        metadata=metadata,
        overlap=overlap,
        calls=existing,
    )
    complete = {
        "schema_version": "ccb-v3-frozen-av-evaluation-complete-v1",
        "completed_at": util.utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "units_sha256": sha256_file(units_path),
        "calls_sha256": sha256_file(calls_path),
        "summary_sha256": sha256_file(args.output_dir / "summary.json"),
        "gold_results_sha256": sha256_file(
            args.output_dir / "gold_results.jsonl"
        ),
        "calls": len(existing),
        "tasks": len(units),
        "gold": len(
            util.read_jsonl(args.output_dir / "gold_results.jsonl")
        ),
        "checks": {
            "task_coverage_exact": len(units) == 328,
            "gold_coverage_exact": len(
                util.read_jsonl(args.output_dir / "gold_results.jsonl")
            )
            == 463,
            "replicate_coverage_exact": len(existing)
            == len(batches) * args.replicates,
            "gates_hidden_from_evaluator": True,
            "no_generation_or_filtering_rerun": True,
        },
    }
    if not all(complete["checks"].values()):
        raise EvaluationError("completion checks failed")
    util.atomic_bytes(complete_path, util.json_bytes(complete))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-actionable", type=Path, required=True)
    parser.add_argument("--v1-variants", type=Path, required=True)
    parser.add_argument("--v1-v3-mapping", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--agent1-gate", type=Path, required=True)
    parser.add_argument("--agent2-v1-gate", type=Path, required=True)
    parser.add_argument("--agent2-v2-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--candidate-batch-size", type=int, default=24)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-live", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvaluationError("evaluation output is active") from exc
        output = execute(args)
    if output is not None:
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
