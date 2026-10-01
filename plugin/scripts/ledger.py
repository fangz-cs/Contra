#!/usr/bin/env python3
"""What has been decided for you in this repository, and is it still true.

A disclosure made during one session is read once and scrolls away. Over weeks
the decisions accumulate — quietly, in code nobody revisits — and no one can
answer the question that matters for a codebase increasingly written by an
assistant: which parts of this were the assistant's call rather than mine?

This reads the verification log the hook appends to and builds that answer. It
also re-checks: a disclosure whose test no longer exists, or whose function has
since been rewritten, is reported as stale rather than silently carried
forward, because an assumption nobody has re-verified is not the same thing as
one that still holds.
"""
from __future__ import annotations

import argparse
import collections

TIER_ORDER = ["beyond_scope", "declined", "high", "medium", "low",
              "housekeeping"]
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hooks"))
import text  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank import MARK, grade  # noqa: E402


def load(log: Path) -> list[dict]:
    rows = []
    if not log.exists():
        return rows
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue          # a torn final line costs one entry, not the ledger
        for item in entry.get("results", []):
            rows.append({**item, "recorded_at": entry.get("at"),
                         "cwd": entry.get("cwd")})
    return rows


def function_changed_since(repo: Path, path: str, at: str | None) -> bool | None:
    """Has the file been committed to since this disclosure was recorded?"""
    if not at:
        return None
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--since", at, "--format=%h", "--", path],
            cwd=str(repo), capture_output=True, text=True, timeout=30).stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return None      # unknown, which the caller already renders as "no flag"
    return bool(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--html", type=Path, help="also write a standalone page here")
    args = ap.parse_args()

    rows = load(args.log)
    if not rows:
        print(text.t("ledger_empty"))
        return

    for r in rows:
        test = r.get("test_path")
        r["_stale"] = False
        if test and not (args.repo / test).exists():
            r["_stale"] = text.t("stale_test_gone")
        elif function_changed_since(args.repo, r.get("file") or '', r.get("recorded_at")):
            r["_stale"] = text.t("stale_file_changed")

    by_file = collections.defaultdict(list)
    for r in rows:
        by_file[r.get("file") or text.t("unknown_file")].append(r)

    tiers = collections.Counter(grade(r) for r in rows)
    verified = sum(1 for r in rows if r.get("verdict") == "confirmed")
    stale = sum(1 for r in rows if r["_stale"])
    print(text.t("ledger_summary", n=len(rows), files=len(by_file)))
    scope_n = tiers["beyond_scope"] + tiers["declined"]
    print(text.t("ledger_tiers", beyond=tiers["beyond_scope"],
                 declined=tiers["declined"], high=tiers["high"],
                 medium=tiers["medium"], low=tiers["low"]))
    # Scope decisions are never settled by a test, so counting them in the
    # denominator would read as a coverage gap that no work could close.
    testable = len(rows) - scope_n
    print(text.t("ledger_verified", verified=verified, testable=testable)
          + (text.t("ledger_stale", stale=stale) if stale else ""))
    print()
    for path, items in sorted(by_file.items(),
                              key=lambda kv: -sum(grade(i) in TIER_ORDER[:3]
                                                  for i in kv[1])):
        print(f"── {path}")
        for i in sorted(items, key=lambda x: TIER_ORDER.index(grade(x))):
            flag = f"  ⟳ {i['_stale']}" if i["_stale"] else ""
            print(f"   {MARK[grade(i)]} {i.get('function')}: {str(i.get('gap'))[:64]}{flag}")
        print()

    if args.html:
        args.html.write_text(_html(by_file, tiers, verified, stale), encoding="utf-8")
        print(text.t("html_written", path=args.html))


def _html(by_file, tiers, verified, stale) -> str:
    total = sum(len(v) for v in by_file.values())
    css = """body{font:14px/1.6 system-ui,sans-serif;max-width:60rem;margin:2rem auto;
padding:0 1.5rem;color:#1a1a1a;background:#fafafa}
h1{font-size:1.3rem;margin:0 0 .3rem}.sub{color:#666;margin-bottom:1.5rem}
.tier{display:inline-block;width:1.2rem}.high{color:#c0392b}.medium{color:#b8860b}
.low{color:#888}.beyond_scope{color:#8e44ad}.declined{color:#2980b9}.file{font-family:ui-monospace,monospace;font-size:.85rem;
color:#555;margin:1.4rem 0 .4rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}
.row{padding:.35rem 0 .35rem .3rem}.gap{font-weight:500}
.meta{color:#777;font-size:.85rem}.stale{color:#c0392b;font-size:.8rem}
@media(prefers-color-scheme:dark){body{background:#161616;color:#e8e8e8}
.file{color:#aaa;border-color:#333}.meta{color:#999}}"""
    parts = [f"<style>{css}</style>",
             f"<h1>{text.t('html_title')}</h1>",
             f"<div class=sub>{text.t('html_sub', n=total, files=len(by_file))} · "
             + text.t("ledger_tiers", beyond=tiers["beyond_scope"],
                      declined=tiers["declined"], high=tiers["high"],
                      medium=tiers["medium"], low=tiers["low"]).strip() + " · "
             + text.t("ledger_verified", verified=verified,
                      testable=total - tiers["beyond_scope"]
                      - tiers["declined"]).strip()
             + (" · " + text.t("ledger_stale", stale=stale).strip()
                if stale else "") + "</div>"]
    for path, items in sorted(by_file.items(),
                              key=lambda kv: -sum(grade(i) in TIER_ORDER[:3]
                                                  for i in kv[1])):
        parts.append(f"<div class=file>{path}</div>")
        for i in sorted(items, key=lambda x: TIER_ORDER.index(grade(x))):
            t = grade(i)
            stale_note = (f"<div class=stale>⟳ {i['_stale']}</div>" if i["_stale"] else "")
            parts.append(
                f"<div class=row><span class='tier {t}'>{MARK[t]}</span>"
                f"<span class=gap>{i.get('gap') or ''}</span>"
                f"<div class=meta>{i.get('function')} · "
                + text.t("html_meta", chosen=i.get("chosen") or "",
                         rival=i.get("rival") or "")
                + f"</div>{stale_note}</div>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
