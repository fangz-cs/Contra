#!/usr/bin/env python3
"""Replay a frozen ordered question plan in the official CCB V3 harness.

Only clarification matching is evaluated. The runner never invokes a
code-generation model; it records one logical terminal turn so the official
TKQR and ORA implementations count the question rounds consistently.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from forkcast.ccb_v3_harness import (
    HarnessError,
    index_tasks,
    load_plan,
    read_jsonl,
    run_question_task,
    sha256_bytes,
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


def repair_and_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    rows: list[dict[str, Any]] = []
    complete_bytes = 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            torn = index == len(lines) - 1 and not line.endswith(b"\n")
            if not torn:
                raise HarnessError(f"{path}: invalid JSONL row {index + 1}")
            with path.open("r+b") as handle:
                handle.truncate(complete_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            break
        if not isinstance(row, dict):
            raise HarnessError(f"{path}: row {index + 1} is not an object")
        rows.append(row)
        complete_bytes += len(line)
    return rows


def bootstrap_release(release: Path):
    if not release.is_dir():
        raise HarnessError(f"ClarifyCodeBench release not found: {release}")
    sys.path.insert(0, str(release))
    from clarifycodebench import interact, prompts  # noqa: PLC0415
    from clarifycodebench.llm import make_client  # noqa: PLC0415

    return interact, prompts, make_client


def release_hashes(release: Path) -> dict[str, str]:
    files = {
        "interact": release / "clarifycodebench/interact.py",
        "llm": release / "clarifycodebench/llm.py",
        "metrics": release / "clarifycodebench/metrics.py",
        "prompts": release / "clarifycodebench/prompts.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise HarnessError(f"release files missing: {missing}")
    return {
        name: sha256_bytes(path.read_bytes()) for name, path in files.items()
    }


def build_manifest(
    *,
    data_path: Path,
    plan_path: Path,
    plan: dict[str, Any],
    release: Path,
    prompts: Any,
    judge_model: str,
    max_turns: int,
    consistency: int,
) -> dict[str, Any]:
    return {
        "schema_version": "ccb-v3-pregenerated-run-manifest-v1",
        "evaluation_scope": "clarification_only",
        "method": plan["method"],
        "judge_model": judge_model,
        "max_turns": max_turns,
        "question_budget": max_turns - 1,
        "consistency": consistency,
        "code_generation_evaluated": False,
        "logical_terminal_turn": True,
        "data_sha256": sha256_bytes(data_path.read_bytes()),
        "plan_sha256": sha256_bytes(plan_path.read_bytes()),
        "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "system_prompt_sha256": sha256_bytes(
            prompts.SYSTEM_PROMPT.encode("utf-8")
        ),
        "judge_prompt_sha256": sha256_bytes(
            prompts.JUDGE_PROMPT.encode("utf-8")
        ),
        "release_hashes": release_hashes(release),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data_v3/ClarifyCodeBench-v3-interactive.jsonl"),
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--release",
        type=Path,
        required=True,
        help="official clarifycodebench_release checkout",
    )
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--consistency", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_turns < 1 or args.consistency < 1 or args.workers < 1:
        raise SystemExit("max-turns, consistency, and workers must be positive")
    interact, prompts, make_client = bootstrap_release(args.release.resolve())
    tasks = index_tasks(read_jsonl(args.data))
    plan = load_plan(args.plan, dataset_path=args.data, tasks=tasks)
    manifest = build_manifest(
        data_path=args.data,
        plan_path=args.plan,
        plan=plan,
        release=args.release.resolve(),
        prompts=prompts,
        judge_model=args.judge_model,
        max_turns=args.max_turns,
        consistency=args.consistency,
    )
    total_questions = int(plan["question_count"])
    omitted = sum(
        max(0, len(entry["questions"]) - (args.max_turns - 1))
        for entry in plan["tasks"].values()
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "tasks": len(tasks),
                    "questions": total_questions,
                    "questions_omitted_by_turn_budget": omitted,
                    "method": plan["method"],
                    "manifest": manifest,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if not args.confirm:
        raise SystemExit("--confirm is required for judge API calls")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(f"refusing resume with manifest drift: {manifest_path}")
    else:
        atomic_json(manifest_path, manifest)

    lock_path = args.out.with_suffix(args.out.suffix + ".lock")
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(
            lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
        )
    except BlockingIOError as exc:
        lock_handle.close()
        raise SystemExit(f"another runner owns {lock_path}") from exc

    existing_rows = repair_and_read(args.out)
    done = {
        str(row["task_id"])
        for row in existing_rows
        if row.get("finished") is True
        and row.get("plan_sha256") == manifest["plan_sha256"]
    }
    todo = [task for task_id, task in tasks.items() if task_id not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    client = make_client(api_key=args.api_key, base_url=args.base_url)

    write_lock = threading.Lock()
    output_handle = args.out.open("a", encoding="utf-8")

    def work(task: dict[str, Any]) -> dict[str, Any]:
        task_id = str(task["task_id"])
        try:
            record = run_question_task(
                interact=interact,
                client=client,
                task=task,
                plan_entry=plan["tasks"][task_id],
                judge_model=args.judge_model,
                max_turns=args.max_turns,
                consistency=args.consistency,
                max_retries=args.max_retries,
            )
        except Exception as exc:  # noqa: BLE001
            record = {
                "task_id": task_id,
                "id": task.get("question_id"),
                "finished": False,
                "failure": f"{type(exc).__name__}: {exc}"[:2000],
                "evaluation_scope": "clarification_only",
                "code_generation_evaluated": False,
            }
        record.update(
            {
                "method": plan["method"],
                "plan_sha256": manifest["plan_sha256"],
                "benchmark_sha256": manifest["data_sha256"],
                "judge_model": args.judge_model,
            }
        )
        return record

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(work, task) for task in todo]
            for future in as_completed(futures):
                record = future.result()
                payload = json.dumps(record, ensure_ascii=False) + "\n"
                with write_lock:
                    output_handle.write(payload)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
    finally:
        output_handle.close()
        lock_handle.close()

    selected_ids = {str(task["task_id"]) for task in todo}
    latest = {
        str(row["task_id"]): row
        for row in repair_and_read(args.out)
        if row.get("plan_sha256") == manifest["plan_sha256"]
    }
    failed = sorted(
        task_id
        for task_id in selected_ids
        if latest.get(task_id, {}).get("finished") is not True
    )
    print(
        json.dumps(
            {
                "already_done": len(done),
                "attempted": len(todo),
                "failed": len(failed),
                "failed_task_ids": failed[:20],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
