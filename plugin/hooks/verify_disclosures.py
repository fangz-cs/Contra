#!/usr/bin/env python3
"""Stop hook: verify the assumptions the assistant just disclosed.

An assistant that discloses its assumptions is only half the mechanism. Some of
those disclosures are wrong -- a fork it thinks exists but does not, a rival
reading that would in fact behave identically. Left unfiltered they cost the
reader more than silence would, because each one asks for attention that turns
out to be unnecessary.

This runs after the assistant stops, not when it decides to. It extracts the
disclosed assumptions, builds the two readings for each, runs them against the
code, and blocks the stop when a disclosure does not hold up -- returning the
verdicts so the report can be corrected. The assistant does not get to skip it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402
import config  # noqa: E402
import text  # noqa: E402

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
TIMEOUT_EXTRACT = config.get("timeout_extract")
TIMEOUT_BUILD = config.get("timeout_build")
# The probe finds ~7 decisions per change where self-report offered ~2, so a
# cap of 3 would now discard most of what it surfaces. Scope decisions are kept
# ahead of implementation ones when the cap bites: they cost no verification and
# they are the ones the reader cannot reconstruct from the code.
MAX_ITEMS = config.get("max_items")
EXTRACT_MODEL = config.get("extract_model")
PROBE_MODEL = config.get("probe_model")
BUILD_MODEL = config.get("build_model")
STATE_DIR = baseline.STATE_DIR
# Set on every `claude -p` this hook starts. Seeing it means we are running
# inside our own child, and going further would start another one.
REENTRY = "CONTRA_INSIDE"
# Reading the repository finds decisions the diff alone cannot show -- a new
# branch that bypasses a fast path the rest of the module uses, a sibling whose
# behaviour no longer agrees. Running commands finds nothing and risks
# everything.
TOOLS = "Read,Grep,Glob"

PROBE_SYSTEM = """You are given a change request, the code as it stood before,
and the diff that was written to satisfy it.

List every decision the diff settles that the request did not state, and say
which kind each one is:

  "beyond_scope"   the diff changed something the request did not name -- a
                   neighbouring function, a test that conflicted with the new
                   behaviour, a helper removed, a parameter added.
  "declined"       the diff left something that the change itself now puts in
                   question: a sibling whose behaviour no longer agrees with it,
                   a guard whose interaction with the new code is unresolved, a
                   behavioural change with nothing pinning it down.
  "implementation" the request left a detail open and the diff picked one:
                   which separator, whether blank counts as empty, which
                   exception type, what to return in a case never mentioned.

Look hardest for the first two. They are easy to miss precisely because the diff
looks tidy: a coherent change is read as a whole, and nobody learns which parts
of it they never asked for.

Exclude, they are not decisions about this change: renaming, formatting, and
project process (no changelog entry, no news file, no version bump). Exclude a
difference in style between two places whose behaviour is identical -- report a
neighbour only when the change leaves its BEHAVIOUR in question.

Judge the request and the diff. Do not treat a choice as settled because it is
customary or because a competent developer would assume it -- reliance on
convention is the thing being surfaced here, and counting it as settled would
hide exactly the cases worth reporting.

Reply with one ```json fenced block:
{"decisions": [{"kind": "beyond_scope"|"declined"|"implementation",
                "file": "path/to/mod.py", "function": "name"|null,
                "gap": "what the request left open, or what was done unasked",
                "context": "the line or expression this is about",
                "chosen": "what the diff does",
                "rival": "what an equally faithful alternative does",
                "scope": "local"|"caller"|"cross_module",
                "caller_hint": "the call site scope refers to, if any",
                "trigger": "common"|"edge",
                "reversible": true|false}]}

`trigger` is "common" when ordinary input reaches the difference and "edge"
when only a boundary case does -- an empty input, a single element, a duplicate,
an absent optional field. Omit the field rather than guessing: it decides
whether the choice is worth the cost of a test.

These three fields were missing here while the other half of the pipeline
produced them, so entries found by review reached the reader with blank lines
where entries the assistant disclosed had detail. Same view, two qualities.
Write `gap`, `chosen`, `rival` and `context` in the same language as the
change request. These go straight into a view the person who wrote that
request reads. Mixing languages across entries also breaks the text
matching that folds the two lists together, so the same decision would
be shown twice.

"""


EXTRACT_SYSTEM = """You are given a change request and the report an assistant
wrote after acting on it. Read the decisions out of the report; the request is
here so you can match its language and see what it did and did not ask for.

Extract every disclosed assumption: a place where the report says it settled
something the request did not specify. For each, give the file and function it
concerns, what the request left open, the reading the assistant implemented, and
the rival reading it named (or your best statement of one if it named none).

Ignore explanations of how the code works, justifications that a choice is
correct, test results, and offers to do further work.

Record what kind of decision each one is. This matters more than anything else
about it, because the three kinds are different problems for the reader:

  "beyond_scope"  the report changed something the request did not name -- a
                  neighbouring function reverted, a conflicting test deleted, a
                  helper removed, a parameter added that nobody asked for
  "declined"      the report judged part of the request out of scope, or left
                  something stale because touching it felt beyond the brief
  "implementation" the request left a detail open and the report picked one:
                  which separator, whether blank counts as empty, which
                  exception type

The first two are about what was done, not how, and they are not settled by any
test -- a change nobody authorised stays unauthorised even when it behaves
sensibly. Only "implementation" needs a discriminating test.

For each one also record three facts about the code, not opinions about
importance:

  "scope"      "local" if the difference is confined to this function's own
               result; "caller" if a listed call site would take a different
               branch, get a different type, or need changing; "cross_module"
               if the change leaves this function disagreeing with a sibling or
               another module.
  "reversible" false when acting on the wrong reading is expensive to undo --
               data written in the chosen format, a message sent, a module-level
               constant other code has already imported, a migration run. true
               when changing your mind means editing this function again.
  "trigger"    "common" if ordinary input reaches the difference; "edge" if
               only a boundary case does -- empty input, a single element, a
               duplicate, an absent optional field. Omit the field rather than
               guessing: it decides whether the choice is worth a test, and a
               guess spends or saves that test for no reason.
  "caller_hint" the call site that scope refers to, if any.

Give "file" as a plain repository-relative path with no quotes around it, and
"function" as the enclosing function's name when there is one ("Class.method"
for a method) or null when the change sits at module level. The name is only
used for labelling; the change itself is located by its text, so a decision in a
constant, an import or a class attribute is as reportable as one in a function.

Reply with one ```json fenced block:
{"items": [{"file": "path/to/mod.py", "function": "name"|null,
            "kind": "beyond_scope"|"declined"|"implementation",
            "gap": "...", "context": "...", "chosen": "...", "rival": "...",
            "scope": "local"|"caller"|"cross_module",
            "trigger": "common"|"edge", "reversible": true|false,
            "caller_hint": "..."}]}
Return {"items": []} if the report discloses no assumption.
Write `gap`, `chosen`, `rival` and `context` in the same language as the
change request. These go straight into a view the person who wrote that
request reads. Mixing languages across entries also breaks the text
matching that folds the two lists together, so the same decision would
be shown twice.

"""

BUILD_SYSTEM = """You are given a Python function as it now stands, and two
readings of an ambiguity in the change that was just made to it.

Produce two things.

1. A pytest test that passes on the CURRENT implementation and would fail if the
   rival reading had been implemented instead. Assert whatever actually carries
   the difference -- it is often not the return value. pytest can observe
   warnings (pytest.warns and its count), raised exceptions, stdout (capsys),
   files, import-time behaviour (importlib.reload), calls (monkeypatch), and
   objects built by fixtures. Pick the surface where these two readings diverge
   and assert against that surface directly.

2. The rival reading, expressed as a patch: a list of exact replacements
   {"old": "<text exactly as it appears now>", "new": "<what it becomes>"}.
   Each "old" must appear EXACTLY ONCE in the file -- include a line or two of
   surrounding context if the snippet alone would be ambiguous -- and must stay
   under 40 lines. Change only what that reading requires; a version that
   accommodates both readings satisfies neither. The change need not be inside
   a function: a constant, an import or a class attribute is a valid target.

The test must import from the real package (e.g. `from pkg.mod import fn`) and
must stand alone in one file.

For each disclosure also say what kind of input its test needed: "common" if it
is the sort of value the listed callers pass in ordinary use, "edge" if it takes
an empty, malformed, or otherwise unusual value to reach the difference.

You may be given several disclosures at once. They share the file sources above.
Treat each independently and return exactly one result per disclosure, in the
order you were given them -- including when there is only one.

Reply with one ```json fenced block and nothing else:
{"results": [{"test_code": "import pytest\n...",
              "rival_edits": [{"old": "...", "new": "..."}],
              "trigger": "common"|"edge"}]}"""


TIMING: list[tuple[str, float]] = []
# What each model call cost, recorded per stage. Latency was measurable from the
# start; spend was not, because the hook's calls are separate `claude -p`
# processes and their cost never appears in the host's own total. A campaign was
# reported at half its real price for exactly that reason.
SPEND: list[dict] = []


def timed(label: str):
    """Record how long a stage took, so optimisation targets the real cost."""
    class _T:
        def __enter__(self):
            self.t = time.monotonic(); return self
        def __exit__(self, *a):
            TIMING.append((label, round(time.monotonic() - self.t, 1)))
    return _T()


def _unwrap(raw: str, label: str, model: str | None) -> str:
    """Pull the reply out of the JSON envelope, recording what it cost.

    Anything that is not the envelope is passed straight through: an older CLI,
    an error path, or a future format change must cost the caller its reply, not
    silently turn a good answer into an empty one.

    `total_cost_usd` covers the whole session, while `usage` describes only its
    last message -- these calls run an agent loop, not a single request, so the
    two are not two views of the same thing and are recorded as they are.
    """
    try:
        env = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
    # Identify the envelope by the CLI's own marker, not by the presence of a
    # "result" key. Every prompt here asks for a fenced JSON reply, and a model
    # is free to name a field "result" -- unwrapping that would hand the caller
    # the field's value in place of the whole answer.
    if not isinstance(env, dict) or env.get("type") != "result":
        return raw
    usage = env.get("usage") or {}
    SPEND.append({
        "stage": label, "model": model or "default",
        "cost_usd": env.get("total_cost_usd"),
        "api_calls": env.get("num_turns"),
        "last_msg_cache_creation": usage.get("cache_creation_input_tokens", 0),
        "last_msg_cache_read": usage.get("cache_read_input_tokens", 0),
        "last_msg_input": usage.get("input_tokens", 0),
        "last_msg_output": usage.get("output_tokens", 0),
    })
    return env.get("result") or ""


def claude(system: str, payload: str, timeout: int, model: str | None = None,
           label: str = "") -> str:
    # Every invocation reloads the assistant's own system prompt -- measured at
    # ~23.5k tokens to answer "ok" -- so the count of calls matters far more
    # than the size of what we send. Hence one batched call below, and a small
    # model for the part that is pure extraction.
    # `--restricted` does three jobs. It drops the tools that run commands and
    # fetch the web, which these calls never need -- they read a request and a
    # diff and answer in JSON -- and must not have, because the diff is the
    # user's repository content arriving as untrusted input. It ignores the
    # settings files, so the child does not inherit this very hook and start a
    # third process. And it cuts the system prompt the child loads from ~24k
    # tokens to ~3.6k, which is most of what each call was paying for.
    #
    # It is incompatible with `--allow-dangerously-skip-permissions`
    # ("bypassPermissions not supported in restricted mode"), so the three tools
    # are allowed by name instead. Without the allow-list every read would
    # prompt, and under `-p` a prompt nobody can answer is a denial.
    cmd = ["claude", "-p", system + "\n\n" + payload,
           "--restricted", "--tools", TOOLS, "--allowed-tools", TOOLS,
           "--disable-slash-commands", "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    env = {**os.environ, REENTRY: "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", timeout=timeout, env=env)
    return _unwrap(proc.stdout, label, model)


def fenced(text: str):
    m = re.search(r"```(?:json)?\s*\n(.*)```", text, re.DOTALL)
    try:
        return json.loads(m.group(1) if m else text)
    except Exception:
        return None


def last_assistant_text(transcript: Path, payload: dict | None = None) -> str:
    """Prefer what the payload hands us; fall back to the file.

    The transcript is written asynchronously, so at the moment a Stop hook runs
    the final assistant message may not be on disk yet. The payload's own copy
    has no such race.
    """
    direct = (payload or {}).get("last_assistant_message")
    if isinstance(direct, str) and direct.strip():
        return direct
    if isinstance(direct, dict):
        content = direct.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text"]
            if any(parts):
                return "\n".join(parts)
    if not transcript.is_file():
        return ""
    text = ""
    for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("type") != "assistant":
            continue
        content = (row.get("message") or {}).get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = [b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text"]
            if any(parts):
                text = "\n".join(parts)
    return text


def probe_diff(request: str, diff: str, cwd: Path) -> list[dict]:
    """Read the diff without reading the assistant's account of it.

    Self-report and third-party review are not the same instrument. An assistant
    describing its own work has reasons not to say "I changed something you did
    not ask for" or "I decided this part was out of scope", and measurement bore
    that out: of the decisions a reviewer found in the diff, roughly a fifth
    appeared in the assistant's own list. The probe has no such reason, and it
    reads the same material.
    """
    if not diff.strip():
        return []
    out = fenced(claude(PROBE_SYSTEM, json.dumps(
        {"request": request[:4000], "diff": diff[:14000]},
        ensure_ascii=False, indent=1), TIMEOUT_BUILD,
        model=PROBE_MODEL, label="probe"))
    found = (out or {}).get("decisions")
    return found if isinstance(found, list) else []


def _similar(a: str, b: str) -> float:
    """Character-bigram overlap, divided by the shorter side.

    Jaccard was the obvious choice and the wrong one: two writers describing the
    same decision rarely spend the same number of words on it, and dividing by
    the union punishes exactly that. On a measured run the true duplicates
    scored 0.21-0.25 by Jaccard, below anything usable, and 0.34-0.59 this way.
    """
    def grams(s: str) -> set:
        s = re.sub(r"\W+", "", (s or "").lower())
        return {s[i:i + 2] for i in range(len(s) - 1)} or {s}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


def merge_disclosures(mine: list[dict], theirs: list[dict]) -> list[dict]:
    """Combine the two lists locally.

    A model call to do this took 65 seconds -- a third of the hook's whole
    latency, and all of it on the reader's critical path. Text overlap does it
    for free, at a known cost: on the measured run it caught three of four
    duplicate pairs, missing one where the two writers shared almost no
    vocabulary. Under-merging shows the reader one entry twice; the 65 seconds
    were worse.

    Entries that survive on both lists are marked "both" rather than folded
    away: agreement between the author and an independent reader is a signal
    about that entry, not redundancy to hide.
    """
    # 0.28 was measured on 132 entries where true pairs scored 0.34+ and the
    # highest false pair 0.14. A later run showed that separation does not
    # hold: two genuine duplicates scored 0.15 and 0.20, while an unrelated
    # pair scored 0.52 on the strength of a shared "Accept-Encoding". Chinese
    # rewordings of one idea can share almost no characters, so this measure
    # is weaker than that first sample suggested.
    #
    # It stays deliberately conservative anyway, because the two errors do not
    # cost the same. Merging two different decisions destroys one of them and
    # nobody can tell afterwards. Failing to merge shows the reader one thing
    # twice, and they can see that for themselves. In the 0.10-0.28 band there
    # are about 1.6 pairs per turn and most of them are NOT duplicates, so
    # lowering the bar would trade a visible annoyance for a silent loss.
    SAME = config.get("merge_threshold")

    def kind_of(item: dict) -> str:
        return item.get("kind") or "implementation"

    out = [{**i, "source": "assistant"} for i in mine]
    for item in theirs:
        match = None
        best = SAME
        for kept in out:
            if (kept.get("file") or "") != (item.get("file") or ""):
                continue
            # A scope decision and an implementation detail are never the same
            # entry however alike they read, and requiring the kinds to agree
            # is what pushes the false pairs clear of the threshold.
            if kind_of(kept) != kind_of(item):
                continue
            score = _similar(kept.get("gap"), item.get("gap"))
            if score >= best:
                best, match = score, kept
        if match is not None:
            match["source"] = "both"
            # Prefer whichever statement of the alternative is fuller.
            if len(item.get("rival") or "") > len(match.get("rival") or ""):
                match["rival"] = item["rival"]
            for field in ("scope", "trigger", "reversible", "function"):
                if not match.get(field):
                    match[field] = item.get(field)
        else:
            out.append({**item, "source": "reviewer"})
    return out


def locate(item: dict, cwd: Path) -> tuple[Path | None, str]:
    """Resolve the disclosure to a file that exists.

    Earlier this also demanded that "function" name a real def, which rejected
    every module-level decision -- a default constant, an import, a class
    attribute. Those are located by their text like any other change, so the
    only thing that has to hold here is that the file is real; whether the
    rival patch applies cleanly is settled later, by applying it.
    """
    raw = (item.get("file") or "").strip().strip("'\"").strip()
    if not raw:
        return None, "no file given"

    # The path comes from a model reading the repository, so it is untrusted
    # input: a suggestive comment in the tree is enough to make it name
    # somewhere else. Resolve it and require that it stay inside the working
    # tree -- both "/etc/passwd" and "../../.ssh/id_rsa" were accepted before.
    try:
        root = cwd.resolve()
        target = (root / raw).resolve()
        target.relative_to(root)
    except (ValueError, OSError):
        return None, f"path outside the working tree: {raw}"
    if target.is_symlink():
        return None, f"refusing to follow a symlink: {raw}"
    if not target.is_file():
        return None, f"file not found: {raw}"
    try:
        target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return None, f"cannot read file: {exc}"
    return target, ""


# Settling a choice means running the code, and only Python has a runner wired
# up here. Everything upstream -- reading the diff, naming the decision, ranking
# it -- already works on any language, so the rest is reported normally and this
# says plainly what is missing rather than dressing it up as "no difference".
RUNNABLE = config.suffixes("runnable_suffixes")


def verify_one(item: dict, cwd: Path, built: dict | None = None) -> dict:
    target, problem = locate(item, cwd)
    if target is None:
        return {**item, "verdict": "unverifiable", "why": problem}
    if target.suffix not in RUNNABLE:
        return {**item, "verdict": "unverifiable",
                "why": text.t("no_runner", suffix=target.suffix)}
    item = {**item, "file": str(target.relative_to(cwd))}
    if not isinstance(built, dict) or not built.get("test_code") or not built.get("rival_edits"):
        return {**item, "verdict": "unverifiable",
                "why": "could not write a discriminating test"}
    item = {**item, "trigger": built.get("trigger") or "edge"}
    generated = {"test_code": built["test_code"], "rival_edits": built["rival_edits"]}
    spec = {"root": str(cwd), "file": item["file"], "function": item.get("function"),
            "test_code": built["test_code"], "rival_edits": built["rival_edits"]}
    try:
        proc = subprocess.run([sys.executable, str(SCRIPTS / "discriminate.py")],
                              input=json.dumps(spec), capture_output=True,
                              text=True, timeout=TIMEOUT_BUILD)
    except subprocess.TimeoutExpired:
        # A test that hangs says nothing about the two readings, and letting the
        # exception out cost an entire turn's report -- including the entries
        # that never needed a test at all.
        return {**item, **generated, "verdict": "unverifiable",
                "why": f"the test did not finish within {TIMEOUT_BUILD}s"}
    except Exception as exc:  # noqa: BLE001
        return {**item, **generated, "verdict": "unverifiable",
                "why": f"{type(exc).__name__}: {exc}"[:200]}
    out = fenced(proc.stdout)
    if out is None and proc.stdout.strip().startswith("{"):
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            # Falling through with the real stderr is worth more than raising:
            # "the sandbox refused to start" is a fixable report, "expecting
            # value at line 1" is not.
            out = None
    if not isinstance(out, dict):
        return {**item, **generated, "verdict": "unverifiable",
                "why": proc.stderr[-200:] or "no verdict"}
    return {**item, **generated, **out}


def _payload_for(items: list[dict], cwd: Path) -> str:
    sources: dict[str, str] = {}
    for item in items:
        path = item.get("file") or ""
        if path and path not in sources:
            f = cwd / path
            if f.is_file():
                sources[path] = f.read_text(encoding="utf-8", errors="replace")[:8000]
    return json.dumps({
        "file_sources": sources,
        "disclosures": [{"file": i.get("file"), "function": i.get("function"),
                         "gap": i.get("gap"),
                         "implemented_reading": i.get("chosen"),
                         "rival_reading": i.get("rival")} for i in items],
    }, ensure_ascii=False, indent=1)


def _as_results(parsed, expected: int) -> list[dict | None] | None:
    """Normalise the shapes a model actually returns.

    It may wrap the list as asked, hand back a bare list, or -- most often with
    a single disclosure -- return one flat object. Insisting on the documented
    shape discarded work that had already been done and paid for.
    """
    if isinstance(parsed, dict):
        inner = parsed.get("results")
        if isinstance(inner, list):
            parsed = inner
        elif parsed.get("test_code"):
            parsed = [parsed]
    if not isinstance(parsed, list) or len(parsed) != expected:
        return None
    return [r if isinstance(r, dict) and r.get("test_code") and r.get("rival_edits")
            else None for r in parsed]


def build_all(items: list[dict], cwd: Path) -> list[dict | None]:
    """One model call for every disclosure, each file's source sent once.

    Falls back to per-item calls when the batched reply cannot be lined up with
    the requests. Batching is an optimisation; it must never be the reason a
    disclosure goes unverified.
    """
    if not items:
        return []
    rows = _as_results(fenced(claude(BUILD_SYSTEM, _payload_for(items, cwd),
                                     TIMEOUT_BUILD, model=BUILD_MODEL,
                                     label="build")), len(items))
    if rows is not None and all(r is not None for r in rows):
        return rows
    if len(items) == 1:
        return rows or [None]
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        singles = list(pool.map(
            lambda i: _as_results(fenced(claude(BUILD_SYSTEM, _payload_for([i], cwd),
                                                TIMEOUT_BUILD, model=BUILD_MODEL,
                                                label="build:retry")), 1), items))
    return [(s[0] if s else None) for s in singles]


KIND_ORDER = ("beyond_scope", "declined", "implementation")


def cap(items: list[dict], limit: int) -> list[dict]:
    """Keep a spread across the three kinds instead of the first `limit`.

    Sorting by kind is right for *showing* the list -- work nobody asked for
    belongs at the top. It was also being used to decide what to drop, and that
    is a different question: with the kinds in that order, a change that raised
    many scope questions spent the whole budget before reaching a single
    implementation choice.

    Measured over 40 real commits: among the turns that hit the cap,
    implementation choices fell to 24% of what was shown (41% among turns that
    did not), and three of them showed none at all -- the same three the recall
    measurement counted as misses. The probe had found them; the cap threw them
    away.

    So take one of each kind in turn. Every kind present keeps a share, and a
    kind with few entries never wastes the places it does not need.
    """
    # Within a kind, take what both instruments raised first. Measured on the
    # same 40 commits: an entry the assistant and the reviewer both reported
    # matched the annotator's fork 47% of the time, against 37% for the
    # assistant alone and 30% for the reviewer alone. Agreement is the best
    # predictor available here, so it decides who keeps a place.
    trust = {"both": 0, "assistant": 1, "reviewer": 2}
    items = sorted(items, key=lambda i: trust.get(i.get("source"), 3))

    buckets: dict[str, list[dict]] = {k: [] for k in KIND_ORDER}
    for item in items:
        buckets.setdefault(item.get("kind") or "implementation", []).append(item)
    out: list[dict] = []
    while len(out) < limit and any(buckets.values()):
        for kind in KIND_ORDER:
            if buckets[kind] and len(out) < limit:
                out.append(buckets[kind].pop(0))
    return out


def _pending(session: str) -> Path:
    return STATE_DIR / (session + ".pending.json")


def _hand_off(session: str, state: dict) -> None:
    """Write the async phase's work list so it can never be read half-written.

    The other process polls for this file's existence. Writing in place would
    let it see the file between creation and the last byte, fail to parse it,
    and drop the whole verification without a word. Rename is atomic.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    final = _pending(session)
    tmp = final.with_suffix(".part")
    with tmp.open("w", encoding="utf-8") as h:
        json.dump(state, h, ensure_ascii=False)
        h.flush(); os.fsync(h.fileno())
    tmp.replace(final)


def _nothing(session: str) -> None:
    """Tell the waiting async phase there is no work, so it exits at once."""
    _hand_off(session, {"verify": [], "reason": "sync half reported nothing"})
    print("{}")


def disclose(payload: dict) -> None:
    """Synchronous phase: say what was decided. No verification.

    This is the half the reader actually waits for, so it holds only the two
    calls that produce the list itself. It renders the view directly rather
    than handing it to the assistant to relay, which is why it cannot be the
    asynchronous half: an async hook's JSON output is never shown to the user.
    """
    cwd = Path(payload.get("cwd") or os.getcwd())
    transcript = Path(payload.get("transcript_path") or "")
    session = payload.get("session_id", "nosession")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    marker = STATE_DIR / (session + ".done")
    if marker.exists():
        marker.unlink(missing_ok=True)
        _nothing(session); return
    report = last_assistant_text(transcript, payload)
    if not report.strip():
        _nothing(session); return

    # Scope everything to this turn. No baseline means no claim: reporting on
    # the whole working tree is what this replaced, so falling back to it
    # would quietly restore the bug.
    snap = baseline.load(session)
    if snap is None:
        _nothing(session); return
    diff = baseline.turn_diff(cwd, snap)
    if not diff.strip():
        # The turn answered a question, or edited something that is not Python.
        _nothing(session); return
    request = (snap.get("prompt") or "").strip()
    with timed("extract+probe"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            # The request comes along only so the extraction can match its
            # language; the decisions themselves are read from the report.
            said = pool.submit(lambda: fenced(claude(
                EXTRACT_SYSTEM,
                "---REQUEST---\n" + request[:2000]
                + "\n---REPORT---\n" + report[:12000],
                TIMEOUT_EXTRACT, model=EXTRACT_MODEL, label="extract")))
            seen = pool.submit(probe_diff, request, diff, cwd)
            disclosed = (said.result() or {}).get("items") or []
            found = seen.result()
    # An entry with nothing to read is not an entry. It used to reach the view
    # as an empty shell -- a bullet, a blank line, and one of the eight places
    # spent -- because a model is free to answer `"gap": null` and that is
    # valid JSON.
    items = [i for i in merge_disclosures(disclosed, found)
             if (i.get("gap") or "").strip()]
    if not items:
        _nothing(session); return

    items = cap([dict(i, kind=(i.get("kind") or "implementation"))
                 for i in items], MAX_ITEMS)

    # Only implementation choices are settled by running code. A change nobody
    # asked for is reportable on its own terms.
    scope_items = [i for i in items if i["kind"] != "implementation"]
    todo = [i for i in items if i["kind"] == "implementation"]

    # Verification is the expensive stage; spend it where the answer changes
    # what the reader would do. A local, edge-only detail sits at the bottom of
    # the view either way, so a test for it buys nothing but latency.
    def worth_testing(i: dict) -> bool:
        # Nothing to spend on a file no runner can execute; it is reported
        # either way, and a test that cannot be run is not evidence.
        if not str(i.get("file") or "").endswith(RUNNABLE):
            return False
        """Skip only what is positively known to be local and edge-only.

        Written the other way round -- test it if the scope is wide or the
        trigger common -- an entry whose fields the model simply failed to fill
        in is dropped without a test and without anyone noticing. Absence of a
        judgement is not a judgement that it does not matter.
        """
        if not i.get("reversible", True):
            return True
        return not (i.get("scope") == "local" and i.get("trigger") == "edge")

    verify = [i for i in todo if worth_testing(i)]
    skipped = [dict(i, verdict="untested",
                    why=text.t("skipped_local_edge"))
               for i in todo if not worth_testing(i)]

    rest = [{**i, "verdict": "scope"} for i in scope_items] + skipped
    if os.environ.get("CONTRA_SYNC_VERIFY"):
        try:
            results = run_verification(cwd, verify)
        except Exception as exc:  # noqa: BLE001
            results = [{**i, "verdict": "unverifiable",
                        "why": f"{type(exc).__name__}: {exc}"[:200]}
                       for i in verify]
        write_log(cwd, results + rest, "sync", diff)
        # If the asynchronous hook is also installed it is polling for the
        # hand-off file right now, and would wait out its full timeout for work
        # that has already been done here. Tell it there is none. Observed
        # adding several minutes to every turn when both were configured.
        _hand_off(session, {"verify": [],
                            "reason": "verification ran in the synchronous half"})
        print(json.dumps({"systemMessage": render_view(results + rest)},
                         ensure_ascii=False))
        return

    shown = rest + [{**i, "verdict": "pending"} for i in verify]
    _hand_off(session, {"cwd": str(cwd), "verify": verify, "timing": TIMING,
                        "spend": SPEND, "rest": rest, "diff": diff})
    print(json.dumps({"systemMessage": render_view(shown)}, ensure_ascii=False))


def run_verification(cwd: Path, todo: list[dict]) -> list[dict]:
    """Build a discriminating test for each open choice and run it."""
    if not todo:
        return []
    inscope = [i for i in todo if locate(i, cwd)[0] is not None]
    with timed(f"build×{len(inscope)}"):
        builds = dict(zip((id(i) for i in inscope), build_all(inscope, cwd)))
    def guarded(i: dict) -> dict:
        try:
            return verify_one(i, cwd, builds.get(id(i)))
        except Exception as exc:  # noqa: BLE001
            return {**i, "verdict": "unverifiable",
                    "why": f"{type(exc).__name__}: {exc}"[:200]}

    with timed(f"discriminate×{len(todo)}"):
        with ThreadPoolExecutor(max_workers=max(1, len(todo))) as pool:
            return list(pool.map(guarded, todo))


def write_log(cwd: Path, results: list[dict], note: str = "",
              diff: str = "") -> None:
    """One line per turn, always -- including turns that verified nothing.

    Staying silent when there was nothing to verify is indistinguishable, from
    outside, from having been killed halfway. An evaluation cannot tell those
    apart after the fact, and the difference is exactly the thing it measures.
    """
    log = STATE_DIR / "verifications.jsonl"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as h:
        h.write(json.dumps({"cwd": str(cwd), "results": results,
                            "timing": TIMING, "spend": SPEND, "note": note,
                            "cost_usd": round(sum(s.get("cost_usd") or 0
                                                  for s in SPEND), 4),
                            "diff": diff[:200000]},
                           ensure_ascii=False) + "\n")
        h.flush(); os.fsync(h.fileno())


def verify_phase(payload: dict) -> None:
    """Asynchronous phase: build a discriminating test for each open choice.

    Runs in the background, so nothing here is on the reader's critical path.
    It wakes the assistant only if there are tests worth keeping in the tree.
    """
    session = payload.get("session_id", "nosession")
    pending = _pending(session)
    # The two Stop hooks fire together; this one waits for the other's list.
    for _ in range(400):
        if pending.exists():
            break
        time.sleep(1)
    else:
        return
    try:
        state = json.loads(pending.read_text(encoding="utf-8"))
    except Exception:
        return
    finally:
        pending.unlink(missing_ok=True)

    todo = state.get("verify") or []
    cwd = Path(state.get("cwd") or payload.get("cwd") or os.getcwd())
    TIMING.extend(tuple(x) for x in state.get("timing") or [])
    # The two halves are separate processes, so the synchronous half's spend
    # arrives here or is never written down at all.
    SPEND.extend(x for x in state.get("spend") or [] if isinstance(x, dict))
    if not todo:
        write_log(cwd, state.get("rest") or [], state.get("reason", "nothing to verify"))
        return

    # Do the work in a session of its own, so losing this process does not lose
    # the results. The host kills the whole process group when the assistant
    # stops, and it only waits when a confirmed test wakes the assistant back
    # up -- so without this the runs that confirmed nothing are exactly the runs
    # that lose their record. Detaching separates the two concerns: the child
    # always lands the data, the parent only carries the wake-up.
    log = STATE_DIR / "verifications.jsonl"
    before = log.stat().st_size if log.is_file() else 0
    pid = os.fork()
    if pid == 0:
        os.setsid()
        try:
            results = run_verification(cwd, todo)
        except Exception as exc:  # noqa: BLE001
            results = [{**i, "verdict": "unverifiable",
                        "why": f"{type(exc).__name__}: {exc}"[:200]} for i in todo]
        write_log(cwd, results + (state.get("rest") or []),
                  diff=state.get("diff", ""))
        os._exit(0)

    # Wait for the child rather than redo its work. If this process is killed
    # first the child carries on regardless; only the wake-up is lost, and a
    # host that has already exited had nowhere to wake up to.
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass
    results = []
    if log.is_file() and log.stat().st_size > before:
        with log.open(encoding="utf-8") as h:
            h.seek(before)
            for line in h:
                try:
                    results = json.loads(line).get("results") or results
                except Exception:
                    continue

    keep = [r for r in results
            if r["verdict"] == "confirmed" and r.get("test_code")]
    if not keep:
        return
    # Exit code 2 is the only channel an async hook has back to the assistant,
    # and stderr is its body. Saving the tests is assistant work anyway, so
    # this is the one part of the flow that belongs on this side.
    (STATE_DIR / (session + ".done")).write_text("1")
    saves = "\n\n".join(
        text.t("test_for", function=r.get("function"), gap=r.get("gap")) + "\n"
        f"```python\n{(r.get('test_code') or '').strip()}\n```"
        for r in keep)
    sys.stderr.write(text.t("save_tests", tests=saves))
    sys.exit(2)


def render_view(results: list[dict]) -> str:
    sys.path.insert(0, str(SCRIPTS))
    try:
        from rank import render          # noqa: PLC0415
        return render(results)
    except Exception as exc:             # noqa: BLE001
        return text.t("render_failed", exc=exc)


def main() -> None:
    # Two locks on the same door, because the cost of it opening is unbounded:
    # a hook installed in settings.json fires for every `claude -p`, including
    # the ones this hook starts, and each of those would start three more.
    # `--restricted` already makes the child ignore settings files; this works
    # even if that flag changes meaning, and it is one comparison.
    if os.environ.get(REENTRY):
        print("{}"); return
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print("{}"); return
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_phase(payload)
    else:
        disclose(payload)


if __name__ == "__main__":
    main()
