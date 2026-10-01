#!/usr/bin/env python3
"""Run a coding-agent harness over ClarifyCodeBench under the CONTRA protocol.

Resume is the default: rerunning the exact same command after any kind of death
picks up where it left off, with no duplicate spend and no duplicate rows.

  # free smoke test, writes to results_stub/, zero API calls
  python3 run_harness.py --stub --models qwen3.8-27b --arms native,ask_permitted

  # what it would cost, no API calls
  python3 run_harness.py --models qwen3.8-27b --arms native --dry-run

  # one real call per model, prints kwargs / reply / how it parsed
  python3 run_harness.py --models qwen3.8-27b --preflight

  # the campaign
  python3 run_harness.py --models qwen3.8-27b --arms native,ask_permitted --confirm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import backends  # noqa: E402
import config  # noqa: E402
import store  # noqa: E402
from episode import build_row, run_episode  # noqa: E402
from judge import Judge  # noqa: E402
from llmio import (CallResult, PermanentError, TransientError, extract_reasoning,  # noqa: E402
                   extract_usage, utcnow, with_retries)
from prompts import ARMS, tag_system_prompt  # noqa: E402

DEFAULT_TASKS = Path("data/benchmark/ClarifyCodeBench-v3-actionable.jsonl")

# Circuit breaker: a misconfiguration looks exactly like bad luck until you plot
# it, so the code decides instead of the operator.
BREAKER_CONSECUTIVE = 10
BREAKER_MIN_CELLS = 50
BREAKER_RATE = 0.20


class Breaker:
    def __init__(self, model: str):
        self.model = model
        self.consecutive = 0
        self.permanent = 0
        self.total = 0
        self.last: dict[str, Any] | None = None
        self.tripped = False
        self._lock = threading.Lock()

    def record(self, *, permanent: bool, detail: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.total += 1
            if permanent:
                self.permanent += 1
                self.consecutive += 1
                self.last = detail
            else:
                self.consecutive = 0
            if self.consecutive >= BREAKER_CONSECUTIVE:
                self.tripped = True
            if self.total >= BREAKER_MIN_CELLS and self.permanent / self.total > BREAKER_RATE:
                self.tripped = True

    def diagnosis(self) -> str:
        d = self.last or {}
        return (
            f"\nCIRCUIT BREAKER: model {self.model!r} stopped after "
            f"{self.permanent}/{self.total} permanent failures "
            f"({self.consecutive} consecutive).\n"
            f"  last status : {d.get('status')}\n"
            f"  last message: {str(d.get('message'))[:300]}\n"
            f"  last body   : {str(d.get('body'))[:600]}\n"
            f"  Continuing past this point only burns money. Fix the cause, then "
            f"rerun with --redo-failed.\n"
        )


def git_commit() -> str:
    for args in (["git", "rev-parse", "HEAD"],):
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            pass
    return "not-a-git-repo"


def load_tasks(path: Path, limit: int | None, offset: int = 0) -> list[dict[str, Any]]:
    tasks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    tasks = tasks[offset:]
    return tasks[:limit] if limit else tasks


def make_chat(model: str, base_url: str | None, api_key: str, stub: bool):
    """Chat callable for the judge and for preflight probes."""
    if stub:
        def _stub_chat(model, messages, max_tokens, **kw) -> CallResult:
            # A stub judge that matches on lexical overlap: enough to drive the
            # loop, and it can never reach the real results directory (R8).
            prompt = messages[0]["content"]
            key = prompt.split("[Statement 1]:")[-1].split("[Statement 2]:")[0].strip().lower()
            got = prompt.split("[Statement 2]:")[-1].split("Decision:")[0].strip().lower()
            kw_ = set(key.split()) - {"the", "a", "of", "is", "to", "should", "what", "do", "you"}
            hit = bool(kw_) and len(kw_ & set(got.split())) / max(1, len(kw_)) > 0.5
            return CallResult(text="yes" if hit else "no", finish_reason="stop",
                              response_id="stub-judge", usage={"prompt_tokens": 200,
                              "completion_tokens": 1, "total_tokens": 201},
                              reasoning_text=None, reasoning_opaque=None, latency_s=0.0)
        return _stub_chat

    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key or "local")

    def _chat(model, messages, max_tokens, **kw) -> CallResult:
        t0 = time.time()
        r = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, **kw)
        msg = r.choices[0].message
        rtext, ropaque = extract_reasoning(msg)
        return CallResult(
            text=msg.content or "", finish_reason=r.choices[0].finish_reason,
            response_id=r.id, usage=extract_usage(r.usage), reasoning_text=rtext,
            reasoning_opaque=ropaque, latency_s=time.time() - t0)
    return _chat


def preflight(models: list[str], stub: bool) -> int:
    """One real call per model. Catches a bad base_url, a bad model id and a
    rejected parameter before the campaign, not 400 cells into it.

    Also answers the two questions you cannot infer from a model's name:
      Q1 does it emit reasoning tokens?
      Q2 does the gateway charge them to the max_tokens budget?
    An empty body with finish_reason="length" means the thinking ate the
    answer's budget -- and in a study that codes whether a model asked, an empty
    turn is data corruption, not attrition.
    """
    rc = 0
    # The tag arm replays SYSTEM_PROMPT verbatim; a drifted copy would silently
    # change what that arm measures.
    from prompts import verify_tag_prompt
    try:
        state = verify_tag_prompt()
        print(f"tag-arm SYSTEM_PROMPT: {state} "
              f"({'checked against clarifycodebench' if state == 'ok' else 'digest only'})")
    except AssertionError as exc:
        print(f"tag-arm SYSTEM_PROMPT: DRIFTED -- {exc}")
        rc = 1

    for key in models:
        cfg = config.MODELS[key]
        print(f"\n=== preflight {key} ({cfg.litellm_model}) ===")
        print(f"  base_url : {cfg.base_url()}")
        print(f"  api_key  : {'set' if cfg.api_key() else 'MISSING'} (${cfg.api_key_env})")
        chat = make_chat(key, cfg.base_url(), cfg.api_key(), stub)
        raw_model = cfg.litellm_model.split("/", 1)[-1] if "/" in cfg.litellm_model else cfg.litellm_model
        kwargs: dict[str, Any] = {k: v for k, v in cfg.extra.items()
                                  if k != "chat_template_kwargs"}
        if "chat_template_kwargs" in cfg.extra:
            kwargs["extra_body"] = {"chat_template_kwargs": cfg.extra["chat_template_kwargs"]}
        print(f"  kwargs   : {json.dumps(kwargs, default=str)}")

        probe = ("Think step by step, then answer. A train leaves at 14:37 and the trip "
                 "takes 2h48m with a 19-minute delay. What time does it arrive? "
                 "Show your reasoning.")
        try:
            res = chat(model=raw_model, messages=[{"role": "user", "content": probe}],
                       max_tokens=64, **kwargs)
        except Exception as exc:  # noqa: BLE001
            from llmio import classify
            err = classify(exc)
            print(f"  FAILED: {type(err).__name__} status={err.status}")
            print(f"  body  : {err.body[:500]}")
            rc = 1
            continue

        reasoning = res.usage.get("reasoning_tokens")
        empty = not (res.text or "").strip()
        print(f"  reply    : {(res.text or '')[:160]!r}")
        print(f"  finish   : {res.finish_reason}")
        print(f"  usage    : {json.dumps({k: v for k, v in res.usage.items() if k != 'raw'})}")
        print(f"  Q1 reasoning model : {'YES' if reasoning else 'no/not reported'}")
        if empty and res.finish_reason == "length":
            print("  Q2 budget accounting: REASONING IS BILLED TO max_tokens.")
            print("     Zero-byte answer with finish_reason=length. Every hard prompt will")
            print(f"     silently come back empty. Raise {key}.max_tokens well above the")
            print("     observed reasoning+answer p99 before running anything.")
            rc = 1
        elif res.finish_reason == "length":
            print("  Q2 budget accounting: reasoning billed separately (answer present).")
        else:
            print("  Q2 budget accounting: probe did not hit the cap; rerun with a smaller")
            print("     max_tokens if you need a decisive answer.")
        if not cfg.calibrated:
            print(f"  NOTE: config.MODELS['{key}'].max_tokens={cfg.max_tokens} is still")
            print("        marked CALIBRATE. Set calibrated=True once measured.")

        # Tool calling is what the whole harness runs on. A deployment without
        # --enable-auto-tool-choice returns 400 here, not halfway through.
        try:
            chat(model=raw_model,
                 messages=[{"role": "user", "content": "List /tmp using the tool."}],
                 max_tokens=64,
                 tools=[{"type": "function", "function": {
                     "name": "execute_bash", "description": "Run a bash command",
                     "parameters": {"type": "object",
                                    "properties": {"command": {"type": "string"}},
                                    "required": ["command"]}}}],
                 tool_choice="auto", **kwargs)
            print("  tool calling      : OK")
        except Exception as exc:  # noqa: BLE001
            from llmio import classify
            err = classify(exc)
            print(f"  tool calling      : FAILED status={err.status}")
            print(f"    {err.body[:300]}")
            print("    OpenHands cannot run without native tool calling. For a vLLM")
            print("    deployment add: --enable-auto-tool-choice --tool-call-parser qwen3_coder")
            rc = 1

    # The judge is half the instrument and it is easy to leave mispointed: a
    # wrong endpoint only shows up on the first turn that actually asks, which
    # may be hundreds of cells in.
    j = config.JUDGE
    print(f"\n=== preflight judge ({j.model}) ===")
    print(f"  base_url : {j.base_url()}")
    print(f"  api_key  : {'set' if j.api_key() else 'MISSING'} (${j.api_key_env})")
    try:
        from judge import Judge
        jd = Judge(make_chat(j.model, j.base_url(), j.api_key(), stub), j.model, j.consistency)
        out = jd.answer("Should a and b be printed on separate lines?",
                        [{"question": "Do you want a and b on separate lines?",
                          "answer": "Yes, one per line."},
                         {"question": "What if the list is empty?", "answer": "Print nothing."}],
                        set())
        print(f"  votes    : {out.calls[0].votes}")
        print(f"  verdict  : hit={out.hit} matched={out.matched!r}")
        if out.hit != 1:
            print("  FAILED: the judge did not match an obviously matching pair.")
            print("          Every question would be scored NOT_RELEVANT.")
            rc = 1
        else:
            print("  judge    : OK")
    except Exception as exc:  # noqa: BLE001
        from llmio import classify
        err = classify(exc)
        print(f"  FAILED: {type(err).__name__} status={err.status}")
        print(f"  body  : {err.body[:400]}")
        rc = 1
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen3.8-27b")
    ap.add_argument("--arms", default="native,ask_permitted")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=None, help="at most N tasks")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N tasks; lets two machines split the "
                         "benchmark with no overlap and no shared filesystem")
    ap.add_argument("--workers", type=int, default=4,
                    help="per-model concurrency; vLLM here runs --max-num-seqs 8")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--harness", default="openhands",
                    choices=("openhands", "claude-code"),
                    help="which agent scaffold runs the task")
    ap.add_argument("--stub", action="store_true", help="canned agent, zero API calls")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, touch nothing")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--confirm", action="store_true",
                    help="required before any real API call")
    ap.add_argument("--redo-failed", action="store_true",
                    help="re-attempt cells that ended failed_permanent")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for m in models:
        if m not in config.MODELS:
            print(f"unknown model {m!r}; known: {list(config.MODELS)}", file=sys.stderr)
            return 2
    for a in arms:
        if a not in ARMS:
            print(f"unknown arm {a!r}; known: {list(ARMS)}", file=sys.stderr)
            return 2

    if args.preflight:
        if not args.stub and not args.confirm:
            print("Refusing to call an API without --confirm.", file=sys.stderr)
            return 2
        return preflight(models, args.stub)

    tasks = load_tasks(args.tasks, args.limit, args.offset)
    run_id = args.run_id or f"{'stub' if args.stub else 'real'}-{time.strftime('%Y%m%dT%H%M%S')}"
    root = store.root_for(args.stub)
    run_dir = root / run_id
    commit = git_commit()
    cfg_blob = json.dumps(config.resolved_config(), sort_keys=True, default=str)
    cfg_digest = hashlib.sha256(cfg_blob.encode()).hexdigest()[:16]

    n_cells = len(tasks) * len(arms) * args.samples * len(models)
    print(f"run_id   : {run_id}")
    print(f"mode     : {'STUB (no API calls)' if args.stub else 'REAL'}")
    print(f"out      : {run_dir}")
    print(f"tasks    : {len(tasks)} (offset {args.offset})  models: {models}  "
          f"arms: {arms}  samples: {args.samples}")
    print(f"cells    : {n_cells}")

    # Resume scan. A cell counts as done only in a terminal state.
    done: dict[tuple[str, str], set[str]] = {}
    total_done = 0
    for m in models:
        for a in arms:
            p = store.rows_path(run_dir, m, a)
            d = store.load_done(p, stub=args.stub)
            if args.redo_failed:
                keep = set()
                for row, _ in store.iter_rows(p):
                    if row and row.get("status") == "ok":
                        keep.add(row["cell_key"])
                d = keep
            done[(m, a)] = d
            total_done += len(d)
    print(f"skipped  : {total_done}   remaining: {n_cells - total_done}")

    if args.dry_run:
        from cost import estimate
        estimate(models, arms, tasks, args.samples, n_cells - total_done)
        return 0

    if not args.stub and not args.confirm:
        print("\nRefusing to call an API without --confirm. Plan printed above; "
              "nothing was spent.")
        return 0

    with store.SingleWriterLock(run_dir):
        return _execute(args, models, arms, tasks, run_dir, run_id, commit,
                        cfg_digest, done, n_cells, total_done)


def _execute(args, models, arms, tasks, run_dir, run_id, commit, cfg_digest,
             done, n_cells, total_done) -> int:
    writer = store.Writer(run_dir).start()
    log_path = run_dir / "run_log.jsonl"
    manifest_path = run_dir / "manifest.json"
    breakers = {m: Breaker(m) for m in models}
    counters = {"ok": 0, "failed_permanent": 0, "failed_transient": 0}
    spend = {"usd": 0.0}
    lock = threading.Lock()
    t0 = time.time()

    def event(**kw) -> None:
        writer.put(log_path, {"at": utcnow(), "run_id": run_id, **kw})

    manifest = {
        "run_id": run_id, "git_commit": commit, "started_at": utcnow(),
        "stub": args.stub, "models": models, "arms": arms,
        "n_tasks": len(tasks), "samples": args.samples, "n_cells": n_cells,
        "tasks_file": str(args.tasks), "harness": args.harness,
        "config_digest": cfg_digest,
        "resolved_config": config.resolved_config(),
        "schema_version": store.SCHEMA_VERSION,
        "argv": sys.argv, "spend_usd": 0.0, "counters": counters,
    }
    store.write_json_atomic(manifest_path, manifest)

    # Cells are ordered TASK-MAJOR, not arm-major: an interrupted run then
    # leaves complete tasks rather than half of every task, and half of every
    # task is usually unanalyzable. Do not "optimize" this into arm-major.
    def cells_for(model: str):
        for task in tasks:
            for arm in arms:
                for s in range(args.samples):
                    ck = store.cell_key(model, arm, task["task_id"], s)
                    if ck in done[(model, arm)]:
                        continue
                    yield model, arm, task, s, ck

    def run_cell(model: str, arm: str, task: dict, sample: int, ck: str) -> None:
        br = breakers[model]
        if br.tripped:
            return
        cfg = config.MODELS[model]
        started = utcnow()
        event(kind="cell_started", cell_key=ck)
        workdir, trace_dir = backends.make_workdir(run_dir, model, arm, task["task_id"], sample)
        request_kwargs = {
            "model": cfg.litellm_model, "base_url": cfg.base_url(),
            "max_output_tokens": cfg.max_tokens, "extra": cfg.extra,
            "native_tool_calling": cfg.native_tool_calling,
            "harness": args.harness,
            "max_iteration_per_run": config.MAX_ITERATIONS_PER_RUN,
            "max_budget_per_run": config.MAX_BUDGET_PER_TASK_USD,
            "max_questions": config.MAX_QUESTIONS, "max_turns": args.max_turns,
            "arm": arm,
        }
        session = None
        attempts: list[dict[str, Any]] = []
        episode = error = None
        status = "failed_transient"
        try:
            if args.stub:
                session = backends.StubSession(
                    task=task, arm=arm, sample=sample, workdir=workdir,
                    max_questions=config.MAX_QUESTIONS)
            elif args.harness == "claude-code":
                cc = config.CLAUDE_CODE
                session = backends.ClaudeCodeSession(
                    model_cfg=cfg, workdir=workdir, trace_dir=trace_dir,
                    claude_bin=cc.claude_bin(), base_url=cc.base_url(),
                    auth_token=cc.auth_token(), model=cc.model,
                    small_model=cc.small_model, timeout_s=cc.timeout_s)
            else:
                with lock:
                    kept = counters.get("_trace_kept", 0)
                    keep_trace = kept < config.AGENT.keep_full_trace_first_n
                    if keep_trace:
                        counters["_trace_kept"] = kept + 1
                session = backends.OpenHandsSession(
                    model_cfg=cfg, workdir=workdir, trace_dir=trace_dir,
                    system_prompt=tag_system_prompt() if arm == "tag" else None,
                    max_iterations=config.MAX_ITERATIONS_PER_RUN,
                    budget_usd=config.MAX_BUDGET_PER_TASK_USD,
                    keep_trace=keep_trace)
            jcfg = config.JUDGE
            judge = Judge(make_chat(jcfg.model, jcfg.base_url(), jcfg.api_key(), args.stub),
                          jcfg.model, jcfg.consistency, jcfg.max_tokens, jcfg.extra)
            episode = run_episode(session=session, task=task, arm=arm, judge=judge,
                                  max_questions=config.MAX_QUESTIONS,
                                  max_turns=args.max_turns)
            status = "ok"
            br.record(permanent=False)
        except PermanentError as exc:
            status = "failed_permanent"
            error = {"type": "PermanentError", "status": exc.status,
                     "message": str(exc), "body": exc.body}
            attempts = [asdict(a) for a in getattr(exc, "attempts", [])]
            br.record(permanent=True, detail=error)
        except TransientError as exc:
            status = "failed_transient"
            error = {"type": "TransientError", "status": exc.status,
                     "message": str(exc), "body": exc.body}
            attempts = [asdict(a) for a in getattr(exc, "attempts", [])]
            br.record(permanent=False)
        except Exception as exc:  # noqa: BLE001
            from llmio import classify
            err = classify(exc)
            status = ("failed_permanent" if isinstance(err, PermanentError)
                      else "failed_transient")
            error = {"type": type(err).__name__, "status": err.status,
                     "message": str(err), "body": err.body}
            br.record(permanent=(status == "failed_permanent"), detail=error)
        finally:
            if session is not None:
                session.close()

        row = build_row(
            cell_key=ck, run_id=run_id, git_commit=commit, stub=args.stub,
            model_key=model, litellm_model=cfg.litellm_model, arm=arm,
            sample=sample, task=task, request_kwargs=request_kwargs,
            episode=episode, status=status, started_at=started, ended_at=utcnow(),
            attempts=attempts, error=error,
            schema_version=store.SCHEMA_VERSION, config_digest=cfg_digest)
        writer.put(store.rows_path(run_dir, model, arm), row)

        with lock:
            counters[status] = counters.get(status, 0) + 1
            u = (episode or {}).get("usage_total") or {}
            spend["usd"] += ((u.get("prompt_tokens") or 0) * cfg.input_cost_per_token
                             + (u.get("completion_tokens") or 0) * cfg.output_cost_per_token
                             + (u.get("judge_prompt_tokens") or 0) * config.JUDGE.input_cost_per_token
                             + (u.get("judge_completion_tokens") or 0) * config.JUDGE.output_cost_per_token)
            n = sum(v for k, v in counters.items() if not k.startswith("_"))
            el = time.time() - t0
            rate = n / el if el else 0
            left = n_cells - total_done - n
            print(f"\r[{n}/{n_cells - total_done}] ok={counters['ok']} "
                  f"perm={counters['failed_permanent']} trans={counters['failed_transient']} "
                  f"{rate:.2f} cell/s ETA {left / rate / 60:.1f}m "
                  f"${spend['usd']:.2f}", end="", flush=True)
            manifest["spend_usd"] = round(spend["usd"], 4)
            manifest["counters"] = {k: v for k, v in counters.items()
                                    if not k.startswith("_")}
            manifest["updated_at"] = utcnow()
            store.write_json_atomic(manifest_path, manifest)
        event(kind="cell_done", cell_key=ck, status=status,
              n_retries=len(attempts))

    print(f"\nwatch it with:\n  ! cd {Path.cwd()} && python3 code/harness/progress.py "
          f"--run {run_dir} --plain\n")

    rc = 0
    try:
        # Models run concurrently with each other: separate endpoints, separate
        # quotas, and a 429 on one must not stall the others.
        with ThreadPoolExecutor(max_workers=len(models)) as outer:
            def drive(model: str) -> None:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for c in cells_for(model):
                        pool.submit(run_cell, *c)
            list(outer.map(drive, models))
    finally:
        writer.close()
        print()
        for m, br in breakers.items():
            if br.tripped:
                print(br.diagnosis(), file=sys.stderr)
                rc = 1
        manifest["ended_at"] = utcnow()
        manifest["counters"] = dict(counters)
        manifest["spend_usd"] = round(spend["usd"], 4)
        store.write_json_atomic(manifest_path, manifest)
        print(f"rows written: {writer.written}  spend: ${spend['usd']:.2f}")
        print(f"manifest    : {manifest_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
