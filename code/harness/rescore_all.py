#!/usr/bin/env python3
"""Rescore every arm with one judge, so the arms stay comparable.

CONTRA and the baselines shipped with hit_dict/hit_questions produced by gpt-4o.
Scoring the harness arm with a different judge and comparing against those
numbers would be meaningless. This rescores ALL of them with whatever judge
config.JUDGE names, writing a parallel set of verdicts and leaving the originals
untouched.

Output is one row per (source, record, asked_question), so the per-record
hit_dict can be rebuilt at any threshold without calling the judge again.

  python3 code/harness/rescore_all.py --workers 8 --confirm
  python3 code/harness/rescore_all.py --report          # rebuild from cache

Resume is the default; rerunning skips finished questions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
import store  # noqa: E402
from judge import Judge  # noqa: E402
from llmio import PermanentError, utcnow  # noqa: E402

BENCH = Path("data/benchmark/ClarifyCodeBench-v3-actionable.jsonl")

# Every source of asked questions that has to end up on the same scale.
SOURCES = [
    ("contra", "results/contra/*/results.jsonl"),
    ("harness", "results/harness/*/rows/*.jsonl"),
]


def load_bench() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with BENCH.open() as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                out[t["task_id"]] = t.get("qa_pairs") or []
    return out


def iter_questions(bench: dict[str, list[dict[str, str]]]) -> Iterator[dict[str, Any]]:
    """Every (source, record, asked question, gold question) that needs a verdict."""
    for source, pattern in SOURCES:
        for path in sorted(Path().glob(pattern)):
            for row, _raw in store.iter_rows(path):
                if row is None:
                    continue
                asked = [a["question"] if isinstance(a, dict) else a
                         for a in (row.get("asked_questions") or [])]
                if not asked:
                    continue
                task_id = row.get("task_id")
                for i, aq in enumerate(asked):
                    for g in bench.get(task_id, []):
                        gq = g["question"]
                        key = hashlib.sha256(f"{aq}||{gq}".encode()).hexdigest()[:16]
                        yield {
                            "cell_key": key,
                            "source": source,
                            "file": str(path),
                            "record_id": row.get("task_id"),
                            "arm": row.get("arm"),
                            "model": row.get("model"),
                            "question_index": i,
                            "asked_question": aq,
                            "gold_question": gq,
                        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/harness/rescored"))
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--report", action="store_true", help="rebuild verdicts from cache only")
    ap.add_argument("--threshold", type=int, default=None,
                    help="yes-votes needed (default: protocol majority)")
    args = ap.parse_args()

    j = config.JUDGE
    out_dir = args.out if not args.stub else Path(str(args.out) + "_stub")
    rows_path = out_dir / "verdicts.jsonl"

    if args.report:
        return report(rows_path, stub=args.stub, consistency=j.consistency,
                      threshold=args.threshold)

    bench = load_bench()
    # Deduplicate: the same (asked, gold) recurs across records and arms, and
    # the verdict does not depend on which record it came from.
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for it in iter_questions(bench):
        if it["cell_key"] in seen:
            continue
        seen.add(it["cell_key"])
        items.append(it)

    print(f"judge   : {j.model} @ {j.base_url()}  votes={j.consistency}")
    print(f"extra   : {json.dumps(j.extra)}")
    print(f"pairs   : {len(items)} unique (asked, gold)")
    print(f"calls   : {len(items) * j.consistency}")
    print(f"out     : {rows_path}")

    done = store.load_done(rows_path, stub=args.stub)
    todo = [it for it in items if it["cell_key"] not in done]
    print(f"skipped : {len(done)}   remaining: {len(todo)}")

    if not todo:
        return report(rows_path, stub=args.stub, consistency=j.consistency,
                      threshold=args.threshold)
    if not args.confirm and not args.stub:
        print("\nRefusing to call an API without --confirm.")
        return 0

    from run_harness import make_chat
    writer = store.Writer(out_dir).start()
    lock = threading.Lock()
    n = [0]
    t0 = time.time()

    def work(it: dict[str, Any]) -> None:
        judge = Judge(make_chat(j.model, j.base_url(), j.api_key(), args.stub),
                      j.model, j.consistency, j.max_tokens, j.extra)
        row = dict(it)
        row.update({"stub": args.stub, "judge_model": j.model,
                    "started_at": utcnow(), "schema_version": store.SCHEMA_VERSION})
        try:
            call = judge._vote(it["gold_question"], it["asked_question"])
            row.update({"status": "ok", "verdict": call.verdict, "votes": call.votes,
                        "yes_votes": sum(1 for v in call.votes
                                         if v.strip().lower().startswith("yes"))})
        except PermanentError as exc:
            row.update({"status": "failed_permanent", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            row.update({"status": "failed_transient", "error": f"{type(exc).__name__}: {exc}"})
        row["ended_at"] = utcnow()
        writer.put(rows_path, row)
        with lock:
            n[0] += 1
            el = time.time() - t0
            rate = n[0] / el if el else 0
            print(f"\r[{n[0]}/{len(todo)}] {rate:.1f}/s "
                  f"ETA {(len(todo)-n[0])/rate/60 if rate else 0:.1f}m", end="", flush=True)

    with store.SingleWriterLock(out_dir):
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(work, todo))
        finally:
            writer.close()
            print()
    return report(rows_path, stub=args.stub, consistency=j.consistency,
                  threshold=args.threshold)


def report(rows_path: Path, *, stub: bool, consistency: int,
           threshold: int | None) -> int:
    """Rebuild per-record hit_dicts from the cached votes, at any threshold."""
    rows = [r for r in store.load_rows(rows_path, stub=stub) if r.get("status") == "ok"]
    if not rows:
        print("no verdicts cached yet")
        return 1
    thr = threshold if threshold is not None else consistency // 2 + 1
    verdict = {r["cell_key"]: (r.get("yes_votes", 0) >= thr) for r in rows}

    bench = load_bench()
    print(f"\n=== rescored with {rows[0]['judge_model']} at >={thr}/{consistency} votes ===")
    by_source: dict[str, dict[str, Any]] = {}
    for it in iter_questions(bench):
        v = verdict.get(it["cell_key"])
        if v is None:
            continue
        key = (it["source"], it.get("model") or "-", it.get("arm") or "-")
        agg = by_source.setdefault(str(key), {"questions": set(), "hits": 0, "records": set()})
        agg["records"].add((it["file"], it["record_id"]))
        qk = (it["file"], it["record_id"], it["question_index"])
        if v:
            if qk not in agg["questions"]:
                agg["hits"] += 1
            agg["questions"].add(qk)
    print(f"  {'source / model / arm':<52} {'records':>8} {'hits':>6}")
    for k, agg in sorted(by_source.items()):
        print(f"  {k:<52} {len(agg['records']):>8} {agg['hits']:>6}")
    print("\n  Rebuild any record's hit_dict from verdicts.jsonl; nothing was overwritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
