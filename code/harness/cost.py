"""No-API cost estimate. Every rate here is a placeholder until preflight and a
pilot have measured the real per-episode token counts on this deployment.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

# PLACEHOLDERS -- the operator must calibrate these from a 10-task pilot.
# S     : system prompt + tools, tokens on every call (OpenHands is ~8-9k)
# DELTA : tokens the history grows per agent iteration
# N     : agent iterations per episode
# O     : output tokens per iteration
PILOT = {
    "S": 9000,
    "DELTA": 700,
    "N": {"native": 12, "ask_permitted": 16, "tag": 3},
    "O": 250,
    "JUDGE_IN": 400,      # per vote
    "JUDGE_OUT": 2,
    "JUDGE_VOTES": 3,
}


def episode_tokens(arm: str) -> tuple[int, int]:
    n = PILOT["N"].get(arm, 12)
    s, d, o = PILOT["S"], PILOT["DELTA"], PILOT["O"]
    # Every call resends the whole conversation: n*S + d*n(n+1)/2.
    return n * s + d * n * (n + 1) // 2, n * o


def estimate(models: list[str], arms: list[str], tasks: list[dict[str, Any]],
             samples: int, remaining_cells: int) -> None:
    print("\n--- cost estimate (NO API CALLS) ---")
    print("  All rates below are PLACEHOLDERS. Calibrate PILOT in cost.py from a")
    print("  10-task pilot before trusting any number here.")
    n_tasks = len(tasks)
    # Questions actually judged bound the judge cost; K is the annotation count.
    total_k = sum(len(t.get("qa_pairs") or []) for t in tasks)
    grand = 0.0
    for m in models:
        cfg = config.MODELS[m]
        sub = 0.0
        print(f"\n  {m}  (${cfg.input_cost_per_token * 1e6:.2f}/M in, "
              f"${cfg.output_cost_per_token * 1e6:.2f}/M out)")
        for a in arms:
            ti, to = episode_tokens(a)
            cells = n_tasks * samples
            cost = cells * (ti * cfg.input_cost_per_token + to * cfg.output_cost_per_token)
            sub += cost
            print(f"    {a:<14} {cells:>5} cells  {ti / 1000:>6.1f}k in / "
                  f"{to / 1000:>4.1f}k out each   ${cost:>9.2f}")
        j = config.JUDGE
        jvotes = total_k * samples * len(arms) * config.MAX_QUESTIONS * PILOT["JUDGE_VOTES"]
        jcost = jvotes * (PILOT["JUDGE_IN"] * j.input_cost_per_token
                          + PILOT["JUDGE_OUT"] * j.output_cost_per_token)
        print(f"    {'judge (max)':<14} {jvotes:>5} votes ceiling            "
              f"        ${jcost:>9.2f}")
        print(f"    {'subtotal':<14}{'':>36}${sub + jcost:>9.2f}")
        grand += sub + jcost
    print(f"\n  TOTAL (all {len(models)} model(s), {remaining_cells} cells left): "
          f"${grand:>.2f}")
    if all(config.MODELS[m].input_cost_per_token == 0 for m in models):
        print("  (self-hosted: the dollar figure is 0 by construction; the number")
        print("   that binds is GPU hours, not dollars)")
    print()
