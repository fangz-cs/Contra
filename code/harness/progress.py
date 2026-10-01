#!/usr/bin/env python3
"""Read-only progress watcher.

Truth comes from the fsynced rows on disk -- the same thing resume trusts --
never from parsing the runner's stdout, so this stays correct even if the
runner's terminal is gone. Killing it is a non-event.

  python3 code/harness/progress.py --run results/harness/<run_id> --plain
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import store  # noqa: E402


def snapshot(run_dir: Path) -> dict:
    man = {}
    mp = run_dir / "manifest.json"
    if mp.exists():
        try:
            man = json.loads(mp.read_text())
        except Exception:
            man = {}
    stub = bool(man.get("stub", str(run_dir).startswith("results_stub")))
    out = {"manifest": man, "stub": stub, "models": {}, "torn": 0,
           "empty_length_turns": 0, "near_cap_turns": 0, "total_turns": 0}
    rows_dir = run_dir / "rows"
    if not rows_dir.exists():
        return out
    for p in sorted(rows_dir.glob("*.jsonl")):
        model, _, arm = p.stem.partition("__")
        d = out["models"].setdefault(model, {})
        a = d.setdefault(arm, {"ok": 0, "failed_permanent": 0, "failed_transient": 0,
                               "asked": 0, "hits": 0})
        for row, _raw in store.iter_rows(p):
            if row is None:
                out["torn"] += 1
                continue
            store.assert_provenance(row, p, stub=stub)
            a[row.get("status", "?")] = a.get(row.get("status", "?"), 0) + 1
            a["asked"] += row.get("n_asked", 0) or 0
            a["hits"] += sum((row.get("hit_dict") or {}).values())
            for t in row.get("turns") or []:
                out["total_turns"] += 1
                # R10 assertion, checked live rather than post-hoc.
                if t.get("finish_reason") == "length":
                    if not (t.get("raw_text") or "").strip():
                        out["empty_length_turns"] += 1
                    else:
                        out["near_cap_turns"] += 1
    return out


def render(s: dict, total_cells: int) -> str:
    lines = []
    for model, arms in sorted(s["models"].items()):
        for arm, c in sorted(arms.items()):
            done = c["ok"] + c["failed_permanent"]
            width = 24
            per_arm_total = max(1, total_cells // max(1, sum(len(a) for a in s["models"].values())))
            filled = min(width, int(width * done / per_arm_total))
            bar = "#" * filled + "-" * (width - filled)
            ask_rate = c["asked"] / max(1, c["ok"])
            lines.append(
                f"  {model:<14} {arm:<14} [{bar}] {done:>4}/{per_arm_total:<4} "
                f"ok={c['ok']} perm={c['failed_permanent']} trans={c['failed_transient']} "
                f"asked/ep={ask_rate:.2f} hits={c['hits']}")
    if s["torn"]:
        lines.append(f"  torn lines: {s['torn']} (those cells will re-run)")
    if s["empty_length_turns"]:
        lines.append(f"  !! EMPTY-BODY finish_reason=length turns: {s['empty_length_turns']} "
                     f"-- reasoning is eating the answer budget; raise max_tokens NOW")
    if s["near_cap_turns"]:
        lines.append(f"  truncated (non-empty) turns: {s['near_cap_turns']} / {s['total_turns']}")
    m = s["manifest"]
    if m:
        lines.append(f"  spend ${m.get('spend_usd', 0)}  commit {str(m.get('git_commit'))[:8]}  "
                     f"{'STUB' if s['stub'] else 'REAL'}")
    return "\n".join(lines) or "  (no rows yet)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--plain", action="store_true",
                    help="append a timestamped block per tick, no screen clear")
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()

    total = 0
    while True:
        try:
            s = snapshot(args.run)
            total = s["manifest"].get("n_cells", total)
            body = render(s, total or 1)
        except store.ProvenanceError as exc:
            body = f"  PROVENANCE ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001
            body = f"  (waiting: {type(exc).__name__}: {exc})"
        stamp = time.strftime("%H:%M:%S")
        if args.plain or args.once:
            print(f"[{stamp}] {args.run}\n{body}", flush=True)
        else:
            print("\033[2J\033[H" + f"[{stamp}] {args.run}\n{body}", flush=True)
        if args.log:
            with args.log.open("a") as f:
                f.write(f"[{stamp}]\n{body}\n")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
