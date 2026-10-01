#!/usr/bin/env python3
"""Grade a verified assumption from facts, never from an opinion.

A model asked how important its own guess was would answer in adjectives, and
those adjectives would be unfalsifiable in exactly the way this whole tool
exists to avoid. So the grade is computed: from whether a test separated the
readings, how far the difference reaches, what it takes to trigger it, and
whether choosing wrong is cheap to undo. Every tier traces back to a fact
someone can check.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hooks"))
import text  # noqa: E402

# The one class of declined entry that is safe to push to the bottom, matched
# by text rather than judged by a model.
#
# A model was asked to make this call three times, with three phrasings, and
# returned 42%, 4% and 0% -- against a hand-counted truth of about 15%. The
# wide version buried real problems (an undeclared brotli dependency, seven
# failing tests); the careful ones missed entries that said "update the
# CHANGELOG" in as many words. A regex is worse at the general question and far
# better at this one: it is stable, it costs nothing, and on 132 real entries it
# was right 7 times out of 8.
#
# It finds about 6% where the truth is 15%, and that under-reach is the safe
# direction: a missed entry is merely shown in its usual place.
LEDGER = re.compile(r"CHANGELOG|CHANGES\.(rst|md)|变更日志|更新日志", re.I)


def housekeeping(item: dict) -> bool:
    if (item.get("kind") or "") != "declined":
        return False
    return bool(LEDGER.search(f"{item.get('gap') or ''} {item.get('chosen') or ''}"))

MARK = {"beyond_scope": "⚑", "declined": "⚐",
        "high": "●", "medium": "◐", "low": "○", "housekeeping": "▫"}
KIND_TITLE = ("beyond_scope", "declined", "housekeeping")
EVIDENCE_MARK = {"confirmed": "✓", "not_separated": "?", "unverifiable": "?",
                 "untested": "·", "pending": "⋯"}


def grade(item: dict) -> str:
    """The kind of decision first, then, for implementation choices, its reach.

    A change nobody asked for is not competing with an implementation detail for
    the reader's attention -- it is a different question, and it outranks every
    detail however well tested. So the kind decides the tier before any of the
    code facts are consulted.
    """
    kind = item.get("kind")
    if housekeeping(item):
        # Worth recording -- it is how a repository stays tidy -- but it is not
        # a decision about what the code does, so it never competes for the top
        # of the list. Measured: this class is half of everything reported and
        # the only class that does not separate underspecified requests from
        # settled ones, because it tracks how many conventions the project has.
        return "housekeeping"
    if kind in ("beyond_scope", "declined"):
        return kind
    verified = item.get("verdict") == "confirmed"
    scope = item.get("scope") or "local"
    trigger = item.get("trigger") or "edge"
    reversible = item.get("reversible", True)

    # Anything whose wrong reading is expensive to undo outranks the rest:
    # the user cannot fix it later by reading the code.
    if not reversible:
        return "high"
    if verified and scope in ("caller", "cross_module") and trigger == "common":
        return "high"
    if verified and (scope in ("caller", "cross_module") or trigger == "common"):
        return "medium"
    if verified:
        return "medium" if scope != "local" else "low"
    # Unverified: reach still matters, but nothing here is high.
    return "medium" if scope in ("caller", "cross_module") else "low"


def _width(s: str) -> int:
    """Display columns, not characters -- a CJK glyph takes two."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def render(items: list[dict]) -> str:
    """One block per disclosure, ordered so the first line is the one to read."""
    order = {"beyond_scope": 0, "declined": 1, "high": 2, "medium": 3,
             "low": 4, "housekeeping": 5}
    ranked = sorted(items, key=lambda i: (order[grade(i)], i.get("function") or ''))
    head = text.headline(len(ranked))
    lines = ["━━ " + head + " " + "━" * max(3, 60 - _width(head) - 4)]
    for item in ranked:
        tier = grade(item)
        lines.append("")
        # Say when the assistant did not raise this itself: a reader who knows
        # which entries came from review reads the whole list differently.
        origin = {"reviewer": text.t("from_reviewer"),
                  "both": text.t("from_both")}.get(item.get("source"), "")
        gap = (item.get("gap") or "").strip()
        if tier in KIND_TITLE:
            head_line = text.t("kind_line", title=text.t("kind_" + tier), gap=gap)
            lines.append(f"{MARK[tier]}  {head_line}{origin}")
        else:
            lines.append(f"{MARK[tier]}  {gap}{origin}")
        if item.get("context"):
            lines.append(f"   {item['context'].strip()}")
        chosen = (item.get("chosen") or "").strip()
        rival = (item.get("rival") or "").strip()
        if tier in KIND_TITLE:
            lines.append(text.t("scope_line", chosen=chosen, rival=rival))
            continue
        lines.append(text.t("choice_line", chosen=chosen, rival=rival))
        reach = {
            "cross_module": text.t("reach_cross_module"),
            "caller": text.t("reach_caller",
                             hint=(item.get("caller_hint") or "")).rstrip(),
            "local": text.t("reach_local"),
        }.get(item.get("scope") or "local")
        # No trigger recorded means nobody judged it. Rendering that as
        # "only on edge input" would state a fact the tool does not have.
        trig = {"common": text.t("trigger_common"),
                "edge": text.t("trigger_edge")}.get(item.get("trigger"))
        prefix = "⚠" if tier == "high" else "·"
        facts = [reach] + ([trig] if trig else [])
        if not item.get("reversible", True):
            facts.append(text.t("irreversible"))
        lines.append(f"   {prefix} " + " · ".join(facts))
        mark = EVIDENCE_MARK.get(item.get("verdict"), "?")
        verdict = item.get("verdict")
        if verdict == "confirmed":
            lines.append(f"   {mark} " + text.t(
                "confirmed", path=item.get("test_path") or text.t("see_below")))
        elif verdict == "pending":
            # The view is rendered before verification runs, and only a
            # confirmed test wakes the assistant to report back. Anything that
            # fails to separate would otherwise leave this line stale forever,
            # so it says where the outcome lands either way.
            lines.append(f"   {mark} " + text.t("pending"))
        elif verdict == "untested":
            lines.append(f"   {mark} " + text.t(
                "untested", why=item.get("why") or text.t("untested_default")))
        elif verdict == "not_separated":
            lines.append(f"   {mark} " + text.t("not_separated"))
        else:
            lines.append(f"   {mark} " + text.t(
                "unverifiable",
                why=item.get("why") or text.t("unverifiable_default")))
    return "\n".join(lines)


if __name__ == "__main__":
    import json, sys
    payload = json.load(sys.stdin)
    print(render(payload if isinstance(payload, list) else payload.get("items", [])))
