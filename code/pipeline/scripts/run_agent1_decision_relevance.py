#!/usr/bin/env python3
"""Run the frozen AV-v4 decision-relevance stage without its superseded Agent 2."""
from __future__ import annotations

import argparse
import fcntl
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import run_av_decision_resolution as base


SCHEMA_MANIFEST = "av-agent1-decision-relevance-manifest-v1"
SCHEMA_RESULT = "av-agent1-decision-relevance-candidate-result-v1"
SCHEMA_COMPLETE = "av-agent1-decision-relevance-complete-v1"
METHOD_VERSION = "av-v4.0-decision-relevance-agent1-only"


def build_manifest(
    *,
    input_path: Path,
    models_config_path: Path,
    config: base.ModelConfig,
    candidates: Sequence[base.Candidate],
    chunks: Sequence[base.Chunk],
    max_candidates: int,
    max_prompt_chars: int,
) -> dict[str, Any]:
    input_sha256 = base.sha256_file(input_path)
    prompt_material = {
        "agent1_core": base.AGENT1_CORE_PROMPT,
        "agent1_batch_protocol": base.AGENT1_BATCH_PROTOCOL,
        "demonstrations": {
            "resolved_specification": base.DEMO_RESOLVED_SPECIFICATION,
            "unresolved_specification": base.DEMO_UNRESOLVED_SPECIFICATION,
            "format_question": base.DEMO_FORMAT_QUESTION,
            "format_hypothesis_a": base.DEMO_FORMAT_A,
            "format_hypothesis_b": base.DEMO_FORMAT_B,
            "invalid_input_question": base.DEMO_INVALID_QUESTION,
            "invalid_input_hypothesis_a": base.DEMO_INVALID_A,
            "invalid_input_hypothesis_b": base.DEMO_INVALID_B,
        },
    }
    return {
        "schema_version": SCHEMA_MANIFEST,
        "method": "Decision-Relevance Classification",
        "method_version": METHOD_VERSION,
        "model": config.name,
        "model_id": config.model_id,
        "model_parameters": {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "workers": config.workers,
        },
        "candidate_count": len(candidates),
        "task_count": len({candidate.group_id for candidate in candidates}),
        "candidate_input_path": str(input_path.resolve()),
        "candidate_input_sha256": input_sha256,
        "candidate_input_file_sha256": input_sha256,
        "candidate_semantic_sha256": base.stable_json_sha256(
            [
                {
                    "candidate_id": candidate.candidate_id,
                    "complete_requirement": candidate.requirement,
                    "clarification_question": candidate.question,
                    "interpretation_a": candidate.hypothesis_a,
                    "interpretation_b": candidate.hypothesis_b,
                }
                for candidate in candidates
            ]
        ),
        "prompt_material": prompt_material,
        "prompt_sha256": base.stable_json_sha256(prompt_material),
        "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "frozen_base_runner_sha256": base.sha256_file(
            Path(base.__file__).resolve()
        ),
        "models_config_path": str(models_config_path.resolve()),
        "models_config_sha256": base.sha256_file(models_config_path),
        "batching": {
            "max_candidates": max_candidates,
            "max_prompt_chars": max_prompt_chars,
            "chunk_count": len(chunks),
            "cross_task_batching": False,
        },
        "downstream_contract": {
            "agent2": "V2 singleton over decision_relevant candidates",
            "behavioral_verifier": (
                "self-generated evidence over decision_relevant candidates"
            ),
        },
        "gold_visible": False,
        "benchmark_metadata_visible": False,
        "programs_visible": False,
        "execution_evidence_visible": False,
    }


def verify_existing_complete(
    output_dir: Path, manifest_path: Path
) -> dict[str, Any] | None:
    complete_path = output_dir / "COMPLETE.json"
    results_path = output_dir / "results.jsonl"
    if not complete_path.exists():
        return None
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        complete.get("schema_version") != SCHEMA_COMPLETE
        or complete.get("manifest_sha256")
        != base.sha256_file(manifest_path)
        or complete.get("results_sha256") != base.sha256_file(results_path)
    ):
        raise base.ReplayError("existing Agent1 completion failed integrity")
    return complete


def execute(args: argparse.Namespace) -> dict[str, Any] | None:
    candidates = base.load_candidates(args.input_jsonl)
    config = base.load_model_config(args.models_config, args.model)
    chunks = base.build_chunks(
        candidates,
        stage="decision_relevance",
        max_candidates=args.max_candidates_per_chunk,
        max_prompt_chars=args.max_prompt_chars,
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = build_manifest(
        input_path=args.input_jsonl,
        models_config_path=args.models_config,
        config=config,
        candidates=candidates,
        chunks=chunks,
        max_candidates=args.max_candidates_per_chunk,
        max_prompt_chars=args.max_prompt_chars,
    )
    base.write_immutable(manifest_path, base.json_bytes(manifest))
    chunks_path = output_dir / "decision_relevance_chunks.jsonl"
    base.write_immutable(
        chunks_path,
        base.jsonl_bytes([base.chunk_record(chunk) for chunk in chunks]),
    )
    if args.prepare_only:
        return None
    if not args.confirm_live:
        raise base.ReplayError("live execution requires --confirm-live")

    existing = verify_existing_complete(output_dir, manifest_path)
    if existing is not None:
        return existing

    chunk_results = base.run_stage(
        chunks=chunks,
        transport=base.OpenAITransport(config),
        output_dir=output_dir,
        workers=config.workers,
    )
    decisions = base.flatten_outputs(chunks, chunk_results)

    positions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    disposition_counts: Counter[str] = Counter()
    for candidate in candidates:
        position = positions[candidate.group_id]
        positions[candidate.group_id] += 1
        selected = decisions[candidate.candidate_id] is True
        disposition = "select" if selected else "exclude"
        disposition_counts[disposition] += 1
        rows.append(
            {
                "schema_version": SCHEMA_RESULT,
                "candidate_id": candidate.candidate_id,
                "candidate_position": position,
                "rank": position + 1,
                "ranking_available": False,
                "decision_relevant": selected,
                "final_disposition": disposition,
                "retained_for_behavioral_verification": selected,
                "gate": "agent1_decision_relevance",
            }
        )

    results_path = output_dir / "results.jsonl"
    base.write_immutable(results_path, base.jsonl_bytes(rows))
    calls_path = output_dir / "calls.jsonl"
    chunk_results_path = output_dir / "decision_relevance_chunk_results.jsonl"
    calls = base.read_jsonl(calls_path, repair_output_journal=True)
    input_sha256 = base.sha256_file(args.input_jsonl)
    complete = {
        "schema_version": SCHEMA_COMPLETE,
        "method_version": METHOD_VERSION,
        "completed_at": base.utc_now(),
        "manifest_sha256": base.sha256_file(manifest_path),
        "candidate_input_sha256": input_sha256,
        "candidate_input_file_sha256": input_sha256,
        "chunks_sha256": base.sha256_file(chunks_path),
        "chunk_results_sha256": base.sha256_file(chunk_results_path),
        "calls_sha256": base.sha256_file(calls_path),
        "results_sha256": base.sha256_file(results_path),
        "candidates": len(rows),
        "final_disposition_counts": {
            "exclude": disposition_counts["exclude"],
            "select": disposition_counts["select"],
        },
        "counts": {
            "candidates": len(rows),
            "tasks": len(positions),
            "decision_relevant_true": disposition_counts["select"],
            "decision_relevant_false": disposition_counts["exclude"],
            "physical_calls": len(calls),
            "protocol_repair_calls": sum(
                int(row.get("protocol_attempt") or 0) > 1 for row in calls
            ),
        },
        "checks": {
            "candidate_coverage_exact": len(rows) == len(candidates),
            "binary_dispositions": sum(disposition_counts.values())
            == len(candidates),
            "selection_matches_decision_relevance": all(
                row["retained_for_behavioral_verification"]
                is row["decision_relevant"]
                for row in rows
            ),
        },
    }
    if not all(complete["checks"].values()):
        raise base.ReplayError("Agent1 completion checks failed")
    base.atomic_bytes(output_dir / "COMPLETE.json", base.json_bytes(complete))
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-candidates-per-chunk", type=int, default=32)
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
            raise base.ReplayError(
                "another Agent1 process owns the run lock"
            ) from exc
        output = execute(args)
    if output is not None:
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
