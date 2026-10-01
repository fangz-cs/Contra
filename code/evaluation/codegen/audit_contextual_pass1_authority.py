#!/usr/bin/env python3
"""Seal and independently audit the contextual pass@1 result authority."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from contextual_pass1_common import (
    CODE_NUDGE,
    METHOD_ORDER,
    MODEL_ORDER,
    ExperimentError,
    atomic_json,
    extract_code,
    model_slug,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentError(message)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    require((root / "QUEUE_COMPLETE").is_file(), "queue is not complete")
    transcript_manifest_path = root / "transcripts" / "MANIFEST.json"
    transcript_manifest = read_json(transcript_manifest_path)
    require(
        isinstance(transcript_manifest, dict)
        and transcript_manifest.get("schema_version")
        == "contextual-pass1-transcript-manifest-v1",
        "invalid transcript manifest",
    )
    require(
        transcript_manifest["protocol"]["maximum_qa_turns"] == 5,
        "transcript protocol is not capped at five Q/A turns",
    )
    require(
        all(transcript_manifest.get("checks", {}).values()),
        "transcript manifest checks failed",
    )

    source_path = root / "inputs" / "livecodebench_v6_sources.jsonl"
    sources = read_jsonl(source_path)
    source_qids = {row.get("question_id") for row in sources}
    require(
        len(sources) == 80 and len(source_qids) == 80 and None not in source_qids,
        "LiveCodeBench source census changed",
    )

    checks: dict[str, bool] = {
        "queue_completed": True,
        "five_qa_turn_maximum": True,
        "source_census_80": True,
    }
    cell_authority: dict[str, Any] = {}
    all_transcript_qids: set[str] = set()
    for model in MODEL_ORDER:
        transcript_record = transcript_manifest["models"][model]
        transcript_path = Path(transcript_record["path"])
        require(
            sha256_file(transcript_path) == transcript_record["sha256"],
            f"{model}: transcript hash changed",
        )
        transcripts = read_jsonl(transcript_path)
        transcript_by_cell = {
            (row.get("method"), row.get("task_id")): row
            for row in transcripts
        }
        require(
            len(transcripts) == 1312
            and len(transcript_by_cell) == 1312,
            f"{model}: transcript cell census changed",
        )
        require(
            Counter(row["method"] for row in transcripts)
            == Counter({method: 328 for method in METHOD_ORDER}),
            f"{model}: transcript method census changed",
        )
        for row in transcripts:
            require(
                0 <= row["question_count"] <= 5
                and row["question_count"]
                <= row["source_question_count"],
                f"{model}/{row['task_id']}: Q/A cap violated",
            )
            require(
                len(row["messages"]) == 3 + 2 * row["question_count"]
                and row["messages"][-1]
                == {"role": "user", "content": CODE_NUDGE},
                f"{model}/{row['task_id']}: message shape changed",
            )
            require(
                stable_hash(row["messages"])
                == row["generation_messages_sha256"],
                f"{model}/{row['task_id']}: message hash changed",
            )
            require(
                not (
                    set(row)
                    & {
                        "qa_pairs",
                        "hit_dict",
                        "hit_questions",
                        "matched_key_questions",
                        "private_test_cases",
                        "public_test_cases",
                    }
                ),
                f"{model}/{row['task_id']}: private fields leaked",
            )
            all_transcript_qids.add(row["question_id"])

        generation_root = root / "generation" / model_slug(model)
        generation_summary_path = generation_root / "SUMMARY.json"
        generation_summary = read_json(generation_summary_path)
        require(
            generation_summary.get("schema_version")
            == "contextual-pass1-generation-summary-v1"
            and generation_summary.get("model") == model,
            f"{model}: invalid generation summary",
        )
        generation_path = Path(generation_summary["results"]["path"])
        require(
            sha256_file(generation_path)
            == generation_summary["results"]["sha256"],
            f"{model}: generation hash changed",
        )
        generations = read_jsonl(generation_path)
        generation_by_cell = {
            (row.get("method"), row.get("task_id")): row
            for row in generations
        }
        require(
            len(generations) == 1312
            and len(generation_by_cell) == 1312
            and set(generation_by_cell) == set(transcript_by_cell),
            f"{model}: generation cell census changed",
        )
        for cell, generation in generation_by_cell.items():
            transcript = transcript_by_cell[cell]
            config = generation["request_config"]
            expected_request_hash = stable_hash(
                {
                    "messages": transcript["messages"],
                    "model": model,
                    "max_tokens": config["max_tokens"],
                    "reasoning_effort": config["reasoning_effort"],
                    "temperature": config["temperature"],
                    "stream": False,
                }
            )
            require(
                generation.get("transport_status") == "ok"
                and generation.get("request_sha256")
                == expected_request_hash
                and generation.get("generation_messages_sha256")
                == transcript["generation_messages_sha256"],
                f"{model}/{cell}: generation request binding failed",
            )
            code, parse_status = extract_code(generation["raw_content"])
            require(
                generation.get("code") == code
                and generation.get("parse_status") == parse_status,
                f"{model}/{cell}: code extraction changed",
            )

        evaluation_root = root / "evaluation" / model_slug(model)
        model_summary = read_json(evaluation_root / "SUMMARY.json")
        require(
            model_summary.get("schema_version")
            == "contextual-livecodebench-pass1-model-v1"
            and model_summary.get("model") == model,
            f"{model}: invalid evaluation summary",
        )
        method_authority: dict[str, Any] = {}
        for method in METHOD_ORDER:
            method_summary_path = evaluation_root / method / "SUMMARY.json"
            method_summary = read_json(method_summary_path)
            per_task_path = Path(method_summary["per_task"]["path"])
            require(
                sha256_file(per_task_path)
                == method_summary["per_task"]["sha256"],
                f"{model}/{method}: per-task hash changed",
            )
            per_task = read_jsonl(per_task_path)
            require(
                len(per_task) == 328
                and len({row["task_id"] for row in per_task}) == 328,
                f"{model}/{method}: evaluation task census changed",
            )
            solved = sum(row.get("passed") is True for row in per_task)
            outcomes = dict(Counter(row["outcome"] for row in per_task))
            require(
                solved == method_summary["solved"]
                and solved == model_summary["methods"][method]["solved"]
                and outcomes == method_summary["outcomes"]
                and abs(method_summary["pass_at_1"] - solved / 328) < 1e-12,
                f"{model}/{method}: evaluation metrics changed",
            )
            require(
                all(
                    not (
                        set(row)
                        & {
                            "input",
                            "inputs",
                            "expected",
                            "outputs",
                            "private_test_cases",
                            "public_test_cases",
                        }
                    )
                    for row in per_task
                ),
                f"{model}/{method}: hidden tests leaked to result rows",
            )
            method_authority[method] = {
                "solved": solved,
                "tasks": 328,
                "pass_at_1": solved / 328,
                "outcomes": outcomes,
                "summary": {
                    "path": str(method_summary_path.resolve()),
                    "sha256": sha256_file(method_summary_path),
                    "identity": method_summary["summary_identity"],
                },
                "per_task": {
                    "path": str(per_task_path.resolve()),
                    "sha256": sha256_file(per_task_path),
                },
            }
        cell_authority[model] = {
            "transcript": {
                "path": str(transcript_path.resolve()),
                "sha256": sha256_file(transcript_path),
            },
            "generation": {
                "path": str(generation_path.resolve()),
                "sha256": sha256_file(generation_path),
                "summary_sha256": sha256_file(generation_summary_path),
            },
            "evaluation_summary": {
                "path": str((evaluation_root / "SUMMARY.json").resolve()),
                "sha256": sha256_file(evaluation_root / "SUMMARY.json"),
                "identity": model_summary["summary_identity"],
            },
            "methods": method_authority,
        }
        checks[f"{model}_1312_generations"] = True
        checks[f"{model}_4x328_evaluations"] = True

    require(
        all_transcript_qids <= source_qids,
        "some transcript tasks have no LiveCodeBench source",
    )
    checks["all_task_tests_covered"] = True

    final_path = root / "final" / "FINAL_SUMMARY.json"
    final = read_json(final_path)
    require(
        final.get("schema_version")
        == "contextual-livecodebench-pass1-final-v1"
        and final.get("complete") is True
        and len(final.get("rows") or []) == 16,
        "final 4x4 summary is incomplete",
    )
    for row in final["rows"]:
        expected = cell_authority[row["model"]]["methods"][row["method"]]
        require(
            row["solved"] == expected["solved"]
            and row["tasks"] == 328
            and abs(row["pass_at_1"] - expected["pass_at_1"]) < 1e-12,
            "final table disagrees with method authority",
        )
    checks["final_4x4_recomputed"] = True

    code_files = sorted((root / "code").glob("*"))
    authority = {
        "schema_version": "contextual-livecodebench-pass1-authority-v1",
        "sealed_at": utc_now(),
        "protocol": transcript_manifest["protocol"],
        "checks": checks,
        "inputs": {
            "transcript_manifest": {
                "path": str(transcript_manifest_path.resolve()),
                "sha256": sha256_file(transcript_manifest_path),
                "identity": transcript_manifest["manifest_identity"],
            },
            "livecodebench_sources": {
                "path": str(source_path.resolve()),
                "sha256": sha256_file(source_path),
                "rows": len(sources),
            },
        },
        "implementation": {
            path.name: sha256_file(path)
            for path in code_files
            if path.is_file()
        },
        "cells": cell_authority,
        "final": {
            "path": str(final_path.resolve()),
            "sha256": sha256_file(final_path),
            "identity": final["final_identity"],
            "table_tsv_sha256": sha256_file(root / "final" / "table.tsv"),
            "table_tex_sha256": sha256_file(root / "final" / "table.tex"),
        },
    }
    require(all(checks.values()), "one or more authority checks failed")
    authority["authority_identity"] = stable_hash(authority)
    output_path = root / "AUTHORITY.json"
    atomic_json(output_path, authority)
    print(
        {
            "authority_identity": authority["authority_identity"],
            "checks": len(checks),
            "cells": 16,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
