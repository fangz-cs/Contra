#!/usr/bin/env python3
"""Replay contrastive Agent 2 while reusing frozen Agent 1 decisions."""
from __future__ import annotations

import argparse
import fcntl
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_av_decision_resolution as base


SCHEMA_MANIFEST = "av-agent2-contrastive-manifest-v1"
SCHEMA_CHUNK_RESULT = "av-agent2-contrastive-chunk-result-v1"
SCHEMA_CALL = "av-agent2-contrastive-call-v1"
SCHEMA_RESULT = "av-agent2-contrastive-candidate-result-v1"
SCHEMA_COMPLETE = "av-agent2-contrastive-complete-v1"
METHOD_VERSION = "agent2-v2.0-contrastive-entailment"

CORE_PROMPT = """\
You are a Contrastive Specification-Resolution Verifier for requirements
clarification in code generation.

Given a task specification S, a clarification question q, and two answer
hypotheses h_A and h_B, determine whether S resolves the choice between them.

Return A only if S semantically entails h_A and makes h_B incompatible. Return
B only if S semantically entails h_B and makes h_A incompatible. If both
hypotheses remain compatible with S, or either judgment requires a
task-specific assumption, return unresolved.

Interpret standard mathematical and programming terms conventionally.
Compatibility is not entailment: behavior consistent with one hypothesis does
not exclude additional behavior allowed by the other. Lexical overlap,
isolated values, and non-discriminating examples are insufficient.

For A or B, quote one or more verbatim spans from S whose conjunction
establishes the decision. For unresolved, return no evidence."""

BATCH_PROTOCOL = """\

The user may provide multiple candidates from the same task specification.
Evaluate each candidate independently; candidates do not provide evidence for
one another.

Return strict JSON only:
{"resolutions": [{"candidate_id": "<provided local ID>",
                  "resolution": "A" | "B" | "unresolved",
                  "evidence_spans": ["<verbatim span from S>", ...]}]}

A or B requires at least one evidence span. Unresolved requires an empty
evidence_spans list. Return exactly one result for every provided candidate ID
and no other IDs."""


class Agent2Error(RuntimeError):
    """A frozen input, protocol, or completion invariant failed."""


def response_schema(chunk: base.Chunk) -> dict[str, Any]:
    local_ids = [item.local_id for item in chunk.items]
    return {
        "type": "object",
        "properties": {
            "resolutions": {
                "type": "array",
                "minItems": len(local_ids),
                "maxItems": len(local_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {
                            "type": "string",
                            "enum": local_ids,
                        },
                        "resolution": {
                            "type": "string",
                            "enum": ["A", "B", "unresolved"],
                        },
                        "evidence_spans": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "candidate_id",
                        "resolution",
                        "evidence_spans",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["resolutions"],
        "additionalProperties": False,
    }


def messages(chunk: base.Chunk) -> list[dict[str, str]]:
    final_user = json.dumps(
        {
            "task_specification": chunk.requirement,
            "candidates": [
                base.candidate_payload(item.local_id, item.candidate)
                for item in chunk.items
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    unresolved = {
        "resolutions": [
            {
                "candidate_id": "q001",
                "resolution": "unresolved",
                "evidence_spans": [],
            }
        ]
    }
    resolved = {
        "resolutions": [
            {
                "candidate_id": "q001",
                "resolution": "A",
                "evidence_spans": [base.DEMO_RESOLUTION_SPAN],
            }
        ]
    }
    return [
        {"role": "system", "content": CORE_PROMPT + BATCH_PROTOCOL},
        {
            "role": "user",
            "content": base.demo_user(
                base.DEMO_UNRESOLVED_SPECIFICATION,
                base.DEMO_FORMAT_QUESTION,
                base.DEMO_FORMAT_A,
                base.DEMO_FORMAT_B,
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                unresolved, ensure_ascii=False, separators=(",", ":")
            ),
        },
        {
            "role": "user",
            "content": base.demo_user(
                base.DEMO_RESOLVED_SPECIFICATION,
                base.DEMO_FORMAT_QUESTION,
                base.DEMO_FORMAT_A,
                base.DEMO_FORMAT_B,
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                resolved, ensure_ascii=False, separators=(",", ":")
            ),
        },
        {"role": "user", "content": final_user},
    ]


def parse_response(
    chunk: base.Chunk, text: str
) -> list[dict[str, Any]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Agent2Error("response is not strict JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"resolutions"}
        or not isinstance(value["resolutions"], list)
    ):
        raise Agent2Error("response must contain only resolutions")
    expected = [item.local_id for item in chunk.items]
    expected_set = set(expected)
    parsed: dict[str, dict[str, Any]] = {}
    fields = {"candidate_id", "resolution", "evidence_spans"}
    for index, row in enumerate(value["resolutions"]):
        if not isinstance(row, dict) or set(row) != fields:
            raise Agent2Error(f"resolutions[{index}]: invalid fields")
        candidate_id = row["candidate_id"]
        resolution = row["resolution"]
        spans = row["evidence_spans"]
        if (
            not isinstance(candidate_id, str)
            or candidate_id not in expected_set
            or candidate_id in parsed
        ):
            raise Agent2Error(f"resolutions[{index}]: invalid candidate ID")
        if resolution not in {"A", "B", "unresolved"}:
            raise Agent2Error(f"resolutions[{index}]: invalid resolution")
        if (
            not isinstance(spans, list)
            or any(not isinstance(span, str) or not span for span in spans)
            or len(set(spans)) != len(spans)
        ):
            raise Agent2Error(
                f"resolutions[{index}]: invalid evidence spans"
            )
        if (resolution == "unresolved") is not (len(spans) == 0):
            raise Agent2Error(
                f"resolutions[{index}]: resolution/evidence mismatch"
            )
        if any(span not in chunk.requirement for span in spans):
            raise Agent2Error(
                f"resolutions[{index}]: evidence is not verbatim in S"
            )
        parsed[candidate_id] = {
            "candidate_id": candidate_id,
            "resolution": resolution,
            "evidence_spans": list(spans),
        }
    if set(parsed) != expected_set:
        raise Agent2Error("response candidate coverage mismatch")
    return [parsed[local_id] for local_id in expected]


def execute_chunk(
    *,
    chunk: base.Chunk,
    transport: base.OpenAITransport,
    calls_path: Path,
    results_path: Path,
    journal_lock: threading.Lock,
) -> dict[str, Any]:
    prompt_messages = messages(chunk)
    schema = response_schema(chunk)
    for protocol_attempt in range(1, base.MAX_PROTOCOL_ATTEMPTS + 1):
        result = transport.call(
            messages=prompt_messages,
            schema=schema,
            schema_name=f"agent2_v2_n{len(chunk.items)}",
        )
        call_row: dict[str, Any] = {
            "schema_version": SCHEMA_CALL,
            "call_id": base.stable_json_sha256(
                {
                    "chunk_id": chunk.chunk_id,
                    "protocol_attempt": protocol_attempt,
                    "at": base.utc_now(),
                }
            )[:24],
            "chunk_id": chunk.chunk_id,
            "chunk_input_sha256": chunk.input_sha256,
            "protocol_attempt": protocol_attempt,
            "candidate_count": len(chunk.items),
            "candidate_ids": [
                item.candidate.candidate_id for item in chunk.items
            ],
            "message_sha256": base.stable_json_sha256(prompt_messages),
            "response_schema_sha256": base.stable_json_sha256(schema),
            "status": result.status,
            "completion_text": result.text,
            "finish_reason": result.finish_reason,
            "error": result.error,
            "retry_reasons": list(result.retry_reasons),
            "latency_seconds": result.latency_seconds,
            "usage": dict(result.usage),
            "provider_metadata": dict(result.provider_metadata),
            "completed_at": base.utc_now(),
        }
        outputs: list[dict[str, Any]] | None = None
        parse_error: str | None = None
        if result.status == "ok" and result.text is not None:
            try:
                outputs = parse_response(chunk, result.text)
            except Agent2Error as exc:
                parse_error = str(exc)
        call_row["parse_error"] = parse_error
        base.append_jsonl(calls_path, call_row, lock=journal_lock)
        if outputs is not None:
            row = {
                "schema_version": SCHEMA_CHUNK_RESULT,
                "chunk_id": chunk.chunk_id,
                "chunk_input_sha256": chunk.input_sha256,
                "group_id": chunk.group_id,
                "ordinal": chunk.ordinal,
                "candidate_count": len(chunk.items),
                "outputs": outputs,
                "logical_call_attempts": protocol_attempt,
                "completed_at": base.utc_now(),
            }
            base.append_jsonl(results_path, row, lock=journal_lock)
            return row
        if result.status != "ok":
            raise Agent2Error(
                f"{chunk.chunk_id}: transport failed: {result.error}"
            )
        if protocol_attempt < base.MAX_PROTOCOL_ATTEMPTS:
            prompt_messages = [
                *prompt_messages,
                {"role": "assistant", "content": result.text or ""},
                {
                    "role": "user",
                    "content": (
                        "The response violated the required protocol: "
                        f"{parse_error}. Return corrected JSON covering "
                        "exactly the provided candidate IDs."
                    ),
                },
            ]
    raise Agent2Error(f"{chunk.chunk_id}: protocol repair exhausted")


def validate_chunk(
    chunk: base.Chunk, row: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        row.get("schema_version") != SCHEMA_CHUNK_RESULT
        or row.get("chunk_id") != chunk.chunk_id
        or row.get("chunk_input_sha256") != chunk.input_sha256
        or row.get("group_id") != chunk.group_id
        or row.get("ordinal") != chunk.ordinal
        or row.get("candidate_count") != len(chunk.items)
    ):
        raise Agent2Error(f"{chunk.chunk_id}: result identity mismatch")
    outputs = row.get("outputs")
    normalized = parse_response(
        chunk,
        json.dumps({"resolutions": outputs}, ensure_ascii=False),
    )
    if normalized != outputs:
        raise Agent2Error(f"{chunk.chunk_id}: noncanonical output order")
    return dict(row)


def load_completed(
    path: Path, chunks: Sequence[base.Chunk]
) -> dict[str, dict[str, Any]]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    completed: dict[str, dict[str, Any]] = {}
    for row in base.read_jsonl(path, repair_output_journal=True):
        chunk_id = row.get("chunk_id")
        if (
            not isinstance(chunk_id, str)
            or chunk_id not in chunk_by_id
            or chunk_id in completed
        ):
            raise Agent2Error(f"{path}: invalid chunk result identity")
        completed[chunk_id] = validate_chunk(chunk_by_id[chunk_id], row)
    return completed


def run_chunks(
    *,
    chunks: Sequence[base.Chunk],
    transport: base.OpenAITransport,
    output_dir: Path,
    workers: int,
) -> dict[str, dict[str, Any]]:
    results_path = output_dir / "chunk_results.jsonl"
    calls_path = output_dir / "calls.jsonl"
    if not calls_path.exists():
        base.atomic_bytes(calls_path, b"")
    if not results_path.exists():
        base.atomic_bytes(results_path, b"")
    completed = load_completed(results_path, chunks)
    pending = [chunk for chunk in chunks if chunk.chunk_id not in completed]
    lock = threading.Lock()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                execute_chunk,
                chunk=chunk,
                transport=transport,
                calls_path=calls_path,
                results_path=results_path,
                journal_lock=lock,
            ): chunk
            for chunk in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            chunk = futures[future]
            try:
                completed[chunk.chunk_id] = future.result()
            except Exception as exc:
                failures.append(
                    f"{chunk.chunk_id}: {type(exc).__name__}: {exc}"
                )
            if index % 25 == 0 or index == len(pending):
                print(
                    f"{base.utc_now()} agent2_v2={len(completed)}/"
                    f"{len(chunks)} failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise Agent2Error(
            f"{len(failures)} chunks failed: {' | '.join(failures[:8])}"
        )
    return load_completed(results_path, chunks)


def load_source(
    *,
    candidates: Sequence[base.Candidate],
    results_path: Path,
    manifest_path: Path,
    complete_path: Path,
    expected_selected: int,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        complete.get("manifest_sha256") != base.sha256_file(manifest_path)
        or complete.get("results_sha256") != base.sha256_file(results_path)
        or complete.get("candidate_input_sha256")
        != manifest.get("candidate_input_sha256")
    ):
        raise Agent2Error("source Agent 1 integrity failed")
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    decisions: dict[str, bool] = {}
    for row in base.read_jsonl(results_path):
        candidate_id = row.get("candidate_id")
        decision = row.get("decision_relevant")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in decisions
            or not isinstance(decision, bool)
        ):
            raise Agent2Error("source Agent 1 row is invalid")
        decisions[candidate_id] = decision
    if (
        set(decisions) != candidate_ids
        or sum(decisions.values()) != expected_selected
    ):
        raise Agent2Error("source Agent 1 coverage mismatch")
    return decisions, manifest, complete


def execute(args: argparse.Namespace) -> dict[str, Any] | None:
    candidates = base.load_candidates(args.candidate_input)
    decisions, source_manifest, source_complete = load_source(
        candidates=candidates,
        results_path=args.source_av_results,
        manifest_path=args.source_av_manifest,
        complete_path=args.source_av_complete,
        expected_selected=args.expected_selected,
    )
    if (
        base.sha256_file(args.candidate_input)
        != source_manifest.get("candidate_input_sha256")
    ):
        raise Agent2Error("candidate input differs from source Agent 1")
    selected = [
        candidate
        for candidate in candidates
        if decisions[candidate.candidate_id]
    ]
    chunks = base.build_chunks(
        selected,
        stage="specification_resolution",
        max_candidates=args.max_candidates,
        max_prompt_chars=args.max_prompt_chars,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = args.output_dir / "chunks.jsonl"
    base.write_immutable(
        chunks_path,
        base.jsonl_bytes([base.chunk_record(chunk) for chunk in chunks]),
    )
    config = base.load_model_config(args.models_config, args.model)
    prompt_material = {
        "core": CORE_PROMPT,
        "batch_protocol": BATCH_PROTOCOL,
        "demonstrations": {
            "unresolved_specification": base.DEMO_UNRESOLVED_SPECIFICATION,
            "resolved_specification": base.DEMO_RESOLVED_SPECIFICATION,
            "question": base.DEMO_FORMAT_QUESTION,
            "hypothesis_A": base.DEMO_FORMAT_A,
            "hypothesis_B": base.DEMO_FORMAT_B,
            "resolved_evidence": [base.DEMO_RESOLUTION_SPAN],
        },
    }
    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "method_version": METHOD_VERSION,
        "method": "Contrastive Specification-Resolution Verification",
        "candidate_count": len(candidates),
        "agent1_selected_candidates": len(selected),
        "chunk_count": len(chunks),
        "model": config.name,
        "model_id": config.model_id,
        "model_parameters": {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "workers": config.workers,
        },
        "prompt_material": prompt_material,
        "prompt_sha256": base.stable_json_sha256(prompt_material),
        "candidate_input_sha256": base.sha256_file(args.candidate_input),
        "source_agent1": {
            "results_sha256": base.sha256_file(args.source_av_results),
            "manifest_sha256": base.sha256_file(args.source_av_manifest),
            "complete_sha256": base.sha256_file(args.source_av_complete),
            "method_version": source_complete.get("method_version"),
        },
        "chunks_sha256": base.sha256_file(chunks_path),
        "retention_rule": (
            "decision_relevant AND resolution == unresolved"
        ),
        "evidence_validation": (
            "A/B requires nonempty unique verbatim spans; unresolved "
            "requires an empty list"
        ),
        "agent1_reused_without_new_calls": True,
        "agent1_output_visible_to_agent2": False,
        "gold_visible": False,
        "benchmark_metadata_visible": False,
        "programs_visible": False,
        "execution_evidence_visible": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    base.write_immutable(manifest_path, base.json_bytes(manifest))
    if args.prepare_only:
        return None
    if not args.confirm_live:
        raise Agent2Error("live replay requires --confirm-live")
    complete_path = args.output_dir / "COMPLETE.json"
    results_path = args.output_dir / "results.jsonl"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            complete.get("manifest_sha256")
            != base.sha256_file(manifest_path)
            or complete.get("results_sha256")
            != base.sha256_file(results_path)
        ):
            raise Agent2Error("existing completion failed integrity")
        return complete

    transport = base.OpenAITransport(config)
    completed = run_chunks(
        chunks=chunks,
        transport=transport,
        output_dir=args.output_dir,
        workers=config.workers,
    )
    resolutions: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        values = completed[chunk.chunk_id]["outputs"]
        for item, value in zip(chunk.items, values, strict=True):
            candidate_id = item.candidate.candidate_id
            if value["candidate_id"] != item.local_id:
                raise Agent2Error("flattened local ID mismatch")
            resolutions[candidate_id] = {
                "resolution": value["resolution"],
                "evidence_spans": value["evidence_spans"],
            }
    if set(resolutions) != {
        candidate.candidate_id for candidate in selected
    }:
        raise Agent2Error("Agent 2 v2 coverage mismatch")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        checked = decisions[candidate_id]
        value = (
            resolutions[candidate_id]
            if checked
            else {"resolution": None, "evidence_spans": []}
        )
        rows.append(
            {
                "schema_version": SCHEMA_RESULT,
                "candidate_id": candidate_id,
                "decision_relevant": decisions[candidate_id],
                "resolution_checked": checked,
                "resolution": value["resolution"],
                "evidence_spans": value["evidence_spans"],
                "retained_for_behavioral_verification": (
                    checked and value["resolution"] == "unresolved"
                ),
            }
        )
    base.atomic_bytes(results_path, base.jsonl_bytes(rows))
    calls_path = args.output_dir / "calls.jsonl"
    chunk_results_path = args.output_dir / "chunk_results.jsonl"
    calls = base.read_jsonl(calls_path, repair_output_journal=True)
    counts = {
        "candidates": len(rows),
        "decision_relevant_true": sum(
            row["decision_relevant"] is True for row in rows
        ),
        "decision_relevant_false": sum(
            row["decision_relevant"] is False for row in rows
        ),
        "resolution_A": sum(row["resolution"] == "A" for row in rows),
        "resolution_B": sum(row["resolution"] == "B" for row in rows),
        "unresolved": sum(
            row["resolution"] == "unresolved" for row in rows
        ),
        "retained_for_behavioral_verification": sum(
            row["retained_for_behavioral_verification"] is True
            for row in rows
        ),
        "physical_calls": len(calls),
        "protocol_repair_calls": sum(
            int(row.get("protocol_attempt") or 0) > 1 for row in calls
        ),
        "http_429_events": sum(
            reason.get("status_code") == 429
            for row in calls
            for reason in row.get("retry_reasons", [])
        ),
    }
    complete = {
        "schema_version": SCHEMA_COMPLETE,
        "method_version": METHOD_VERSION,
        "completed_at": base.utc_now(),
        "manifest_sha256": base.sha256_file(manifest_path),
        "candidate_input_sha256": base.sha256_file(args.candidate_input),
        "chunks_sha256": base.sha256_file(chunks_path),
        "chunk_results_sha256": base.sha256_file(chunk_results_path),
        "calls_sha256": base.sha256_file(calls_path),
        "results_sha256": base.sha256_file(results_path),
        "counts": counts,
        "checks": {
            "candidate_coverage_exact": len(rows) == len(candidates),
            "agent1_gate_exact": len(resolutions) == len(selected),
            "resolution_total_exact": (
                counts["resolution_A"]
                + counts["resolution_B"]
                + counts["unresolved"]
                == len(selected)
            ),
            "evidence_contract_exact": all(
                (
                    row["resolution"] in {"A", "B"}
                    and bool(row["evidence_spans"])
                )
                or (
                    row["resolution"] in {"unresolved", None}
                    and not row["evidence_spans"]
                )
                for row in rows
            ),
            "all_evidence_verbatim": all(
                span in candidate.requirement
                for row, candidate in zip(rows, candidates, strict=True)
                for span in row["evidence_spans"]
            ),
            "no_new_agent1_calls": True,
        },
    }
    if not all(complete["checks"].values()):
        raise Agent2Error("completion checks failed")
    base.atomic_bytes(complete_path, base.json_bytes(complete))
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--source-av-results", type=Path, required=True)
    parser.add_argument("--source-av-manifest", type=Path, required=True)
    parser.add_argument("--source-av-complete", type=Path, required=True)
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--expected-selected",
        type=int,
        required=True,
        help="number of candidates selected by the supplied Agent 1 run",
    )
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--max-prompt-chars", type=int, default=60000)
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
            raise Agent2Error("Agent 2 v2 output is active") from exc
        output = execute(args)
    if output is not None:
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
