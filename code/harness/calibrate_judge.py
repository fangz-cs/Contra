#!/usr/bin/env python3
"""Calibrate a candidate judge against the gpt-4o verdicts already on disk.

Why this exists: CONTRA and the baselines were scored by gpt-4o with a 3-vote
majority. Scoring the harness arm with a different judge would make the hit
rates incomparable. But swapping the judge is fine **as long as every arm is
rescored with the same one** -- so the question is only whether a free local
judge agrees with gpt-4o closely enough to stand in.

The labels come from the shipped results: each CONTRA record says which of its
asked questions matched (hit_dict) and which gold question it matched
(hit_questions), which reconstructs (asked_question, gold_question, verdict)
triples at no cost.

  python3 code/harness/calibrate_judge.py --limit 300 --workers 8 --confirm

Resume is the default: rerun the same command and finished pairs are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
import store  # noqa: E402
from judge import Judge  # noqa: E402
from llmio import PermanentError, TransientError, utcnow  # noqa: E402

BENCH = Path("data/benchmark/ClarifyCodeBench-v3-actionable.jsonl")
CONTRA_GLOB = "results/contra/*/results.jsonl"


def build_pairs(limit: int | None) -> list[dict[str, Any]]:
    """Reconstruct (asked, gold, gpt4o_verdict) triples from the shipped runs."""
    bench: dict[str, list[dict[str, str]]] = {}
    with BENCH.open() as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                bench[t["task_id"]] = t.get("qa_pairs") or []

    pairs: list[dict[str, Any]] = []
    for path in sorted(Path().glob(CONTRA_GLOB)):
        backbone = path.parent.name
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                asked = [a["question"] if isinstance(a, dict) else a
                         for a in (r.get("asked_questions") or [])]
                hits = set(r.get("hit_questions") or [])
                hd = r.get("hit_dict") or {}
                golds = [q["question"] for q in bench.get(r["task_id"], [])]
                for i, aq in enumerate(asked):
                    verdict_i = hd.get(str(i))
                    if verdict_i is None:
                        continue
                    for g in golds:
                        label = bool(verdict_i == 1 and g in hits)
                        pid = hashlib.sha256(
                            f"{aq}||{g}".encode()).hexdigest()[:16]
                        pairs.append({
                            "pair_id": pid,
                            "backbone": backbone,
                            "task_id": r["task_id"],
                            "asked_question": aq,
                            "gold_question": g,
                            "gpt4o_verdict": label,
                        })
    # Deduplicate: the same (asked, gold) can recur across backbones.
    seen, uniq = set(), []
    for p in pairs:
        if p["pair_id"] in seen:
            continue
        seen.add(p["pair_id"])
        uniq.append(p)
    # Stratify so a --limit sample keeps both classes.
    pos = [p for p in uniq if p["gpt4o_verdict"]]
    neg = [p for p in uniq if not p["gpt4o_verdict"]]
    if limit:
        half = max(1, limit // 2)
        uniq = pos[:half] + neg[:limit - min(half, len(pos))]
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="stratified sample size")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/harness/judge_calibration"))
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()

    pairs = build_pairs(args.limit)
    j = config.JUDGE
    out_dir = args.out if not args.stub else Path(str(args.out) + "_stub")
    rows_path = out_dir / "pairs.jsonl"

    print(f"judge     : {j.model}  @ {j.base_url()}")
    print(f"votes     : {j.consistency} per pair, max_tokens={j.max_tokens}")
    print(f"kwargs    : {json.dumps(j.extra)}")
    print(f"pairs     : {len(pairs)}  "
          f"(pos {sum(1 for p in pairs if p['gpt4o_verdict'])} / "
          f"neg {sum(1 for p in pairs if not p['gpt4o_verdict'])})")
    print(f"calls     : {len(pairs) * j.consistency}")
    print(f"out       : {rows_path}")

    done = store.load_done(rows_path, stub=args.stub)
    todo = [p for p in pairs if p["pair_id"] not in done]
    print(f"skipped   : {len(done)}   remaining: {len(todo)}")

    if not args.confirm and not args.stub:
        print("\nRefusing to call an API without --confirm.")
        return 0

    from run_harness import make_chat
    writer = store.Writer(out_dir).start()
    lock = threading.Lock()
    n_done = [0]
    t0 = time.time()

    def work(p: dict[str, Any]) -> None:
        judge = Judge(make_chat(j.model, j.base_url(), j.api_key(), args.stub),
                      j.model, j.consistency, j.max_tokens, j.extra)
        row = dict(p)
        row.update({"stub": args.stub, "cell_key": p["pair_id"],
                    "started_at": utcnow(), "schema_version": store.SCHEMA_VERSION})
        try:
            call = judge._vote(p["gold_question"], p["asked_question"])
            row.update({"status": "ok", "judge_verdict": call.verdict,
                        "votes": call.votes, "usage": call.usage})
        except PermanentError as exc:
            row.update({"status": "failed_permanent", "error": str(exc), "body": exc.body})
        except Exception as exc:  # noqa: BLE001
            row.update({"status": "failed_transient", "error": f"{type(exc).__name__}: {exc}"})
        row["ended_at"] = utcnow()
        writer.put(rows_path, row)
        with lock:
            n_done[0] += 1
            el = time.time() - t0
            rate = n_done[0] / el if el else 0
            print(f"\r[{n_done[0]}/{len(todo)}] {rate:.1f}/s "
                  f"ETA {(len(todo)-n_done[0])/rate/60 if rate else 0:.1f}m", end="", flush=True)

    with store.SingleWriterLock(out_dir):
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(work, todo))
        finally:
            writer.close()
            print()
    return report(rows_path, stub=args.stub)


def report(rows_path: Path, *, stub: bool) -> int:
    rows = [r for r in store.load_rows(rows_path, stub=stub) if r.get("status") == "ok"]
    if not rows:
        print("no completed pairs yet")
        return 1
    tp = sum(1 for r in rows if r["gpt4o_verdict"] and r["judge_verdict"])
    fp = sum(1 for r in rows if not r["gpt4o_verdict"] and r["judge_verdict"])
    fn = sum(1 for r in rows if r["gpt4o_verdict"] and not r["judge_verdict"])
    tn = sum(1 for r in rows if not r["gpt4o_verdict"] and not r["judge_verdict"])
    n = len(rows)
    agree = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # Cohen's kappa: agreement above chance, which is what matters when the
    # classes are this unbalanced.
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 0.0

    print(f"\n=== agreement with gpt-4o over {n} pairs ===")
    print(f"                gpt-4o yes   gpt-4o no")
    print(f"  judge yes     {tp:>9}   {fp:>9}")
    print(f"  judge no      {fn:>9}   {tn:>9}")
    print(f"\n  agreement : {agree:.3f}")
    print(f"  precision : {prec:.3f}   recall: {rec:.3f}   F1: {f1:.3f}")
    print(f"  kappa     : {kappa:.3f}")
    if kappa >= 0.8:
        print("\n  Substantial agreement. Safe to rescore every arm with this judge,")
        print("  reporting the agreement figure alongside.")
    else:
        print("\n  NOT close enough to stand in for gpt-4o. Rescoring every arm with")
        print("  it would change the numbers in ways the agreement table cannot")
        print("  account for. Look at the disagreements before using it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
