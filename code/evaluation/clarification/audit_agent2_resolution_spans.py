#!/usr/bin/env python3
"""Blindly audit whether Agent 2 spans directly resolve candidate questions."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SYSTEM_PROMPT = """\
You are auditing extractive evidence for specification resolution.

For each item, decide whether the quoted resolution span, by itself, directly
selects exactly one of answer hypothesis A and answer hypothesis B.

Direct selection requires the explicit content of the span to entail one
hypothesis and exclude the other. Use ordinary compositional meaning of words
present in the span. Do not infer an unstated relation from the span's section
or proximity to other text, generalize from an example, import a task
convention, or choose an answer merely because it is more plausible. A bare
number, symbol, label, or technical term does not establish an omitted
task-specific relation by itself.

Return `direct_A` or `direct_B` only when the span meets that standard.
Otherwise return `not_direct`. Classify the main evidential dependence and
give one concise reason. Evaluate every item independently."""

VERDICTS = frozenset({"direct_A", "direct_B", "not_direct"})
DEPENDENCES = frozenset(
    {
        "explicit_proposition",
        "lexical_definition",
        "local_context",
        "example_generalization",
        "task_convention",
        "insufficient_fragment",
        "other",
    }
)
TRANSIENT_HTTP_STATUSES = frozenset(
    {408, 409, 425, 429, 499, 500, 502, 503, 504, 520, 521, 522, 524, 529}
)
MAX_TRANSPORT_RETRIES = 3
MAX_PROTOCOL_ATTEMPTS = 3
FULL_SPECIFICATION_VISIBLE = False


class AuditError(RuntimeError):
    pass


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise AuditError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def index_unique(
    rows: Iterable[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in output:
            raise AuditError(f"{label}: invalid or duplicate {key}")
        output[value] = row
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ).encode("utf-8")


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise AuditError(f"{path}: immutable artifact drift")
        return
    atomic_bytes(path, payload)


def repair_torn_tail(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        return
    start = raw.rfind(b"\n") + 1
    try:
        complete = isinstance(json.loads(raw[start:]), dict)
    except (UnicodeDecodeError, json.JSONDecodeError):
        complete = False
    with path.open("r+b") as handle:
        if complete:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
        else:
            handle.truncate(start)
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(
    path: Path,
    row: Mapping[str, Any],
    *,
    lock: threading.Lock | None = None,
) -> None:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"

    def write() -> None:
        repair_torn_tail(path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    if lock is None:
        write()
    else:
        with lock:
            write()


def diverse_sample(
    candidate_ids: Sequence[str],
    *,
    cluster_for: Mapping[str, str],
    size: int,
) -> list[str]:
    ordered = sorted(
        candidate_ids,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    selected: list[str] = []
    seen_clusters: set[str] = set()
    for candidate_id in ordered:
        cluster = cluster_for[candidate_id]
        if cluster not in seen_clusters:
            selected.append(candidate_id)
            seen_clusters.add(cluster)
            if len(selected) == size:
                return selected
    for candidate_id in ordered:
        if candidate_id not in selected:
            selected.append(candidate_id)
            if len(selected) == size:
                return selected
    if len(selected) != size:
        raise AuditError("insufficient candidates for diverse sample")
    return selected


def build_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = index_unique(
        read_jsonl(args.candidate_input), "candidate_id", "candidate input"
    )
    metadata = index_unique(
        read_jsonl(args.candidate_metadata),
        "candidate_id",
        "candidate metadata",
    )
    av = index_unique(
        read_jsonl(args.av_results), "candidate_id", "AV results"
    )
    if set(candidates) != set(metadata) or set(candidates) != set(av):
        raise AuditError("candidate, metadata, and AV coverage differ")
    gold_filtered: list[str] = []
    not_relevant_filtered: list[str] = []
    for row in read_jsonl(args.qg_evaluation):
        item_id = str(row["item_id"])
        for question in row["questions"]:
            candidate_id = item_id + "::" + str(question["root_id"])
            result = av[candidate_id]
            filtered_by_agent2 = (
                result["decision_relevant"] is True
                and isinstance(result["resolution_span"], str)
            )
            if not filtered_by_agent2:
                continue
            if question["hit"] == 1:
                gold_filtered.append(candidate_id)
            elif (
                question["hit"] == 0
                and "not relevant to the current task"
                in str(question["judge_reply"]).lower()
            ):
                not_relevant_filtered.append(candidate_id)
    if len(gold_filtered) != args.expected_gold_filtered:
        raise AuditError("Agent-2-filtered Gold census mismatch")
    cluster_for = {
        candidate_id: str(metadata[candidate_id]["question_id"])
        for candidate_id in candidates
    }
    negative_sample = diverse_sample(
        not_relevant_filtered,
        cluster_for=cluster_for,
        size=len(gold_filtered),
    )
    grouped = [
        *[(candidate_id, "gold_filtered") for candidate_id in gold_filtered],
        *[
            (candidate_id, "not_relevant_filtered")
            for candidate_id in negative_sample
        ],
    ]
    grouped.sort(
        key=lambda pair: hashlib.sha256(pair[0].encode("utf-8")).hexdigest()
    )
    items: list[dict[str, Any]] = []
    for index, (candidate_id, group) in enumerate(grouped, 1):
        candidate = candidates[candidate_id]
        span = av[candidate_id]["resolution_span"]
        if not isinstance(span, str) or span not in candidate[
            "complete_requirement"
        ]:
            raise AuditError(f"{candidate_id}: invalid resolution span")
        items.append(
            {
                "schema_version": "agent2-span-audit-item-v1",
                "blind_id": f"x{index:04d}",
                "candidate_id": candidate_id,
                "source_cluster": cluster_for[candidate_id],
                "analysis_group": group,
                "clarification_question": candidate[
                    "clarification_question"
                ],
                "answer_hypothesis_A": candidate["interpretation_a"],
                "answer_hypothesis_B": candidate["interpretation_b"],
                "resolution_span": span,
            }
        )
    return items


def public_item(item: Mapping[str, Any]) -> dict[str, str]:
    return {
        "blind_id": str(item["blind_id"]),
        "clarification_question": str(item["clarification_question"]),
        "answer_hypothesis_A": str(item["answer_hypothesis_A"]),
        "answer_hypothesis_B": str(item["answer_hypothesis_B"]),
        "resolution_span": str(item["resolution_span"]),
    }


def response_schema(blind_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "assessments": {
                "type": "array",
                "minItems": len(blind_ids),
                "maxItems": len(blind_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "blind_id": {
                            "type": "string",
                            "enum": list(blind_ids),
                        },
                        "verdict": {
                            "type": "string",
                            "enum": sorted(VERDICTS),
                        },
                        "evidential_dependence": {
                            "type": "string",
                            "enum": sorted(DEPENDENCES),
                        },
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "blind_id",
                        "verdict",
                        "evidential_dependence",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["assessments"],
        "additionalProperties": False,
    }


def parse_response(
    text: str, blind_ids: Sequence[str]
) -> list[dict[str, str]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AuditError("audit response is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"assessments"}:
        raise AuditError("audit response root is invalid")
    assessments = value["assessments"]
    if not isinstance(assessments, list):
        raise AuditError("assessments is not a list")
    expected = set(blind_ids)
    parsed: dict[str, dict[str, str]] = {}
    for row in assessments:
        if not isinstance(row, dict) or set(row) != {
            "blind_id",
            "verdict",
            "evidential_dependence",
            "reason",
        }:
            raise AuditError("assessment fields are invalid")
        blind_id = row["blind_id"]
        verdict = row["verdict"]
        dependence = row["evidential_dependence"]
        reason = row["reason"]
        if (
            not isinstance(blind_id, str)
            or blind_id not in expected
            or blind_id in parsed
            or verdict not in VERDICTS
            or dependence not in DEPENDENCES
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise AuditError("assessment value is invalid")
        parsed[blind_id] = {
            "blind_id": blind_id,
            "verdict": verdict,
            "evidential_dependence": dependence,
            "reason": reason,
        }
    if set(parsed) != expected:
        raise AuditError("assessment coverage mismatch")
    return [parsed[blind_id] for blind_id in blind_ids]


def call_batch(
    *,
    batch_id: str,
    round_index: int,
    items: Sequence[Mapping[str, Any]],
    client: Any,
    model: str,
) -> dict[str, Any]:
    blind_ids = [str(item["blind_id"]) for item in items]
    schema = response_schema(blind_ids)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"items": [public_item(item) for item in items]},
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]
    transport_errors: list[dict[str, Any]] = []
    protocol_errors: list[str] = []
    started = time.monotonic()
    for protocol_attempt in range(1, MAX_PROTOCOL_ATTEMPTS + 1):
        response = None
        for transport_attempt in range(1, MAX_TRANSPORT_RETRIES + 2):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=8192,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": f"agent2_span_audit_n{len(items)}",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                transport_errors.append(
                    {
                        "attempt": transport_attempt,
                        "status_code": status_code,
                        "error": f"{type(exc).__name__}: {exc}"[:8000],
                    }
                )
                permanent = (
                    isinstance(status_code, int)
                    and 400 <= status_code < 500
                    and status_code not in TRANSIENT_HTTP_STATUSES
                )
                if permanent or transport_attempt > MAX_TRANSPORT_RETRIES:
                    raise AuditError(
                        f"{batch_id}: transport failed: {exc}"
                    ) from exc
                time.sleep(
                    min(30.0, (2 ** (transport_attempt - 1)) * 0.75)
                )
        if response is None or not response.choices:
            raise AuditError(f"{batch_id}: provider returned no choices")
        choice = response.choices[0]
        text = getattr(choice.message, "content", None) or ""
        try:
            assessments = parse_response(text, blind_ids)
        except AuditError as exc:
            protocol_errors.append(str(exc))
            if protocol_attempt >= MAX_PROTOCOL_ATTEMPTS:
                raise
            messages.extend(
                [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            f"Protocol error: {exc}. Return corrected JSON "
                            "covering exactly the supplied blind IDs."
                        ),
                    },
                ]
            )
            continue
        usage = getattr(response, "usage", None)
        raw_usage = usage.model_dump() if hasattr(usage, "model_dump") else {}
        return {
            "schema_version": "agent2-span-audit-call-v1",
            "call_key": f"{batch_id}::round_{round_index}",
            "batch_id": batch_id,
            "round_index": round_index,
            "blind_ids": blind_ids,
            "input_sha256": stable_json_sha256(
                [public_item(item) for item in items]
            ),
            "model": model,
            "temperature": 0.0,
            "assessments": assessments,
            "transport_errors": transport_errors,
            "protocol_errors": protocol_errors,
            "usage": raw_usage,
            "latency_seconds": round(time.monotonic() - started, 3),
            "completed_at": utc_now(),
        }
    raise AssertionError("unreachable")


def majority(values: Sequence[str]) -> str:
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    return value if count >= len(values) // 2 + 1 else "no_majority"


def section_for(requirement: str, span: str) -> str:
    position = requirement.find(span)
    before = requirement[:position].lower()
    markers = [
        ("sample output", "sample_output"),
        ("sample input", "sample_input"),
        ("constraints", "constraints"),
        ("output", "output"),
        ("input", "input"),
    ]
    best = (-1, "body")
    for marker, name in markers:
        found = before.rfind(marker)
        if found > best[0]:
            best = (found, name)
    return best[1]


def render_report(output: Mapping[str, Any]) -> str:
    lines = [
        "# Blind audit of Agent 2 resolution spans",
        "",
        (
            "The evaluator saw only q, h_A, h_B, and the extracted span. "
            "Gold/negative labels, task IDs, the full specification, and "
            "Agent 2's decision were hidden."
        ),
        "",
        "## Majority verdicts",
        "",
        "| Group | n | direct A/B | not direct | no majority |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in ["gold_filtered", "not_relevant_filtered"]:
        row = output["group_summary"][group]
        lines.append(
            f"| {group.replace('_', ' ')} | {row['n']} "
            f"| {row['direct']} ({100 * row['direct'] / row['n']:.1f}%) "
            f"| {row['not_direct']} ({100 * row['not_direct'] / row['n']:.1f}%) "
            f"| {row['no_majority']} |"
        )
    lines.extend(["", "## Evidential dependence", ""])
    for group in ["gold_filtered", "not_relevant_filtered"]:
        lines.append(f"**{group.replace('_', ' ')}**")
        for name, count in output["group_summary"][group][
            "dependence_counts"
        ].items():
            lines.append(f"- {name}: {count}")
        lines.append("")
    lines.extend(
        [
            "## Deterministic span diagnostics",
            "",
            json.dumps(
                output["span_diagnostics"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
        ]
    )
    return "\n".join(lines)


def execute(args: argparse.Namespace) -> dict[str, Any] | None:
    items = build_items(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items_path = args.output_dir / "audit_items.jsonl"
    write_immutable(items_path, jsonl_bytes(items))
    manifest = {
        "schema_version": "agent2-span-audit-manifest-v1",
        "method": "blind three-replicate extractive-evidence audit",
        "model": args.model,
        "temperature": 0.0,
        "replicates": args.replicates,
        "batch_size": args.batch_size,
        "item_count": len(items),
        "group_counts": dict(
            sorted(Counter(item["analysis_group"] for item in items).items())
        ),
        "labels_visible_to_evaluator": False,
        "full_specification_visible_to_evaluator": (
            FULL_SPECIFICATION_VISIBLE
        ),
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": stable_json_sha256(SYSTEM_PROMPT),
        "candidate_input_sha256": sha256_file(args.candidate_input),
        "candidate_metadata_sha256": sha256_file(
            args.candidate_metadata
        ),
        "av_results_sha256": sha256_file(args.av_results),
        "qg_evaluation_sha256": sha256_file(args.qg_evaluation),
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest_path = args.output_dir / "manifest.json"
    write_immutable(manifest_path, json_bytes(manifest))
    if args.prepare_only:
        return None
    if not args.confirm_live:
        raise AuditError("live audit requires --confirm-live")
    complete_path = args.output_dir / "COMPLETE.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            complete.get("manifest_sha256") != sha256_file(manifest_path)
            or complete.get("results_sha256")
            != sha256_file(args.output_dir / "results.jsonl")
        ):
            raise AuditError("existing audit completion failed integrity")
        return complete

    base_url = os.environ.get("CONTRA_BASE_URL")
    api_key = os.environ.get("CONTRA_API_KEY")
    if not base_url or not api_key:
        raise AuditError("set CONTRA_BASE_URL and CONTRA_API_KEY")
    import openai

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=600.0,
        max_retries=0,
    )
    batches = [
        items[index : index + args.batch_size]
        for index in range(0, len(items), args.batch_size)
    ]
    tasks = [
        (
            f"b{batch_index:04d}",
            round_index,
            batch,
        )
        for round_index in range(args.replicates)
        for batch_index, batch in enumerate(batches)
    ]
    calls_path = args.output_dir / "calls.jsonl"
    if not calls_path.exists():
        atomic_bytes(calls_path, b"")
    repair_torn_tail(calls_path)
    existing_rows = read_jsonl(calls_path)
    completed: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        call_key = row.get("call_key")
        if (
            not isinstance(call_key, str)
            or call_key in completed
            or row.get("schema_version") != "agent2-span-audit-call-v1"
        ):
            raise AuditError("invalid or duplicate audit call journal")
        completed[call_key] = row
    pending = [
        task
        for task in tasks
        if f"{task[0]}::round_{task[1]}" not in completed
    ]
    lock = threading.Lock()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                call_batch,
                batch_id=batch_id,
                round_index=round_index,
                items=batch,
                client=client,
                model=args.model,
            ): (batch_id, round_index)
            for batch_id, round_index, batch in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            batch_id, round_index = futures[future]
            try:
                row = future.result()
                append_jsonl(calls_path, row, lock=lock)
                completed[row["call_key"]] = row
            except Exception as exc:
                failures.append(
                    f"{batch_id}/r{round_index}: {type(exc).__name__}: {exc}"
                )
            if index % 10 == 0 or index == len(pending):
                print(
                    f"{utc_now()} audit_calls={len(completed)}/"
                    f"{len(tasks)} failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise AuditError(
            f"{len(failures)} audit calls failed: {' | '.join(failures[:8])}"
        )

    votes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in completed.values():
        for assessment in row["assessments"]:
            votes[str(assessment["blind_id"])].append(assessment)
    item_by_blind = {str(item["blind_id"]): item for item in items}
    results: list[dict[str, Any]] = []
    for blind_id, item in item_by_blind.items():
        item_votes = votes[blind_id]
        if len(item_votes) != args.replicates:
            raise AuditError(f"{blind_id}: replicate coverage mismatch")
        results.append(
            {
                "schema_version": "agent2-span-audit-result-v1",
                "blind_id": blind_id,
                "candidate_id": item["candidate_id"],
                "source_cluster": item["source_cluster"],
                "analysis_group": item["analysis_group"],
                "majority_verdict": majority(
                    [str(vote["verdict"]) for vote in item_votes]
                ),
                "majority_evidential_dependence": majority(
                    [
                        str(vote["evidential_dependence"])
                        for vote in item_votes
                    ]
                ),
                "votes": item_votes,
            }
        )
    results_path = args.output_dir / "results.jsonl"
    atomic_bytes(results_path, jsonl_bytes(results))

    group_summary: dict[str, Any] = {}
    for group in ["gold_filtered", "not_relevant_filtered"]:
        group_rows = [
            row for row in results if row["analysis_group"] == group
        ]
        verdicts = Counter(row["majority_verdict"] for row in group_rows)
        dependencies = Counter(
            row["majority_evidential_dependence"] for row in group_rows
        )
        group_summary[group] = {
            "n": len(group_rows),
            "direct": verdicts["direct_A"] + verdicts["direct_B"],
            "not_direct": verdicts["not_direct"],
            "no_majority": verdicts["no_majority"],
            "verdict_counts": dict(sorted(verdicts.items())),
            "dependence_counts": dict(sorted(dependencies.items())),
        }

    candidates = index_unique(
        read_jsonl(args.candidate_input), "candidate_id", "candidate input"
    )
    span_diagnostics: dict[str, Any] = {}
    for group in ["gold_filtered", "not_relevant_filtered"]:
        group_items = [
            item for item in items if item["analysis_group"] == group
        ]
        lengths = [
            len(re.findall(r"\w+", str(item["resolution_span"])))
            for item in group_items
        ]
        sections = Counter(
            section_for(
                str(candidates[str(item["candidate_id"])][
                    "complete_requirement"
                ]),
                str(item["resolution_span"]),
            )
            for item in group_items
        )
        sorted_lengths = sorted(lengths)
        span_diagnostics[group] = {
            "word_length_mean": sum(lengths) / len(lengths),
            "word_length_median": sorted_lengths[len(lengths) // 2],
            "word_length_le_5": sum(length <= 5 for length in lengths),
            "word_length_le_10": sum(length <= 10 for length in lengths),
            "section_counts": dict(sorted(sections.items())),
        }
    output = {
        "schema_version": "agent2-span-audit-summary-v1",
        "completed_at": utc_now(),
        "group_summary": group_summary,
        "span_diagnostics": span_diagnostics,
        "calls": len(completed),
        "items": len(items),
        "replicates": args.replicates,
        "hashes": {
            "manifest_sha256": sha256_file(manifest_path),
            "items_sha256": sha256_file(items_path),
            "calls_sha256": sha256_file(calls_path),
            "results_sha256": sha256_file(results_path),
        },
    }
    atomic_bytes(args.output_dir / "summary.json", json_bytes(output))
    atomic_bytes(
        args.output_dir / "summary.md",
        render_report(output).encode("utf-8"),
    )
    complete = {
        "schema_version": "agent2-span-audit-complete-v1",
        "completed_at": utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "items_sha256": sha256_file(items_path),
        "calls_sha256": sha256_file(calls_path),
        "results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(args.output_dir / "summary.json"),
        "items": len(items),
        "calls": len(completed),
        "checks": {
            "balanced_groups": all(
                value["n"] == args.expected_gold_filtered
                for value in group_summary.values()
            ),
            "replicate_coverage_exact": all(
                len(votes[str(item["blind_id"])]) == args.replicates
                for item in items
            ),
            "labels_hidden_from_evaluator": True,
        },
    }
    if not all(complete["checks"].values()):
        raise AuditError("audit completion checks failed")
    atomic_bytes(complete_path, json_bytes(complete))
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--av-results", type=Path, required=True)
    parser.add_argument("--qg-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--expected-gold-filtered", type=int, default=139)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-live", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / ".run.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AuditError("audit output directory is active") from exc
        output = execute(args)
    if output is not None:
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
