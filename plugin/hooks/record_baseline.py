#!/usr/bin/env python3
"""UserPromptSubmit hook: mark where this turn starts.

Everything the Stop hook reports is scoped to what happened after this point.
Without it the reviewer reads the whole working tree and calls all of it the
assistant's work.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402
import text  # noqa: E402

# A prompt the harness injected is not a change request, and letting one
# overwrite the real request leaves the reviewer judging the diff against a
# wake-up notice. Observed doing exactly that: a background-task notification
# was recorded as the request for a whole evaluation cell.
AUTHORED = {"user", "sdk", None}

# `source` alone did not catch it, so the text is checked too. These are the
# shapes the harness wraps its own messages in; a person writing a change
# request does not open with one.
INJECTED = ("<task-notification>", "<system-reminder>", "<command-name>",
            "<local-command-stdout>", "<bash-input>", "Caveat:")


def authored(payload: dict, text: str) -> bool:
    """Whether this prompt is a request a person made, not one the harness sent."""
    if payload.get("source") not in AUTHORED:
        return False
    return not text.lstrip().startswith(INJECTED)


def main() -> None:
    # A `claude -p` started by the Stop hook submits a prompt too, and would
    # snapshot the tree on its way in. Nothing reads that snapshot and the scan
    # is not free, so stop before doing the work.
    if os.environ.get("CONTRA_INSIDE"):
        print("{}"); return
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print("{}"); return
    cwd = Path(payload.get("cwd") or os.getcwd())
    session = payload.get("session_id", "nosession")

    prompt = (payload.get("prompt") or "").strip()
    previous = baseline.load(session) or {}
    if not authored(payload, prompt) and previous:
        # A harness message interrupting a turn does not start a new one. Moving
        # the baseline here was wrong twice over: the request being worked on is
        # still the old one, and resetting the snapshot after the assistant has
        # already edited makes the turn's diff come out empty -- so the whole
        # report is lost, silently, on exactly the long conversations where a
        # notification is most likely to arrive.
        print("{}")
        return

    try:
        snap = baseline.snapshot(cwd)
    except Exception as exc:  # noqa: BLE001
        # A failed snapshot must not cost the user their turn. Stop will find
        # no baseline, report nothing, and say so rather than guessing.
        print(json.dumps({"suppressOutput": True,
                          "systemMessage": text.t("baseline_failed", exc=exc)}))
        return
    snap["prompt"] = prompt
    snap["cwd"] = str(cwd)
    # Recorded so a wrong request can be traced back to how it arrived.
    snap["source"] = payload.get("source")
    snap["authored"] = authored(payload, (payload.get("prompt") or "").strip())
    baseline.save(session, snap)
    print("{}")


if __name__ == "__main__":
    main()
