"""Agent-session backends.

The episode loop (episode.py) is written against ``AgentSession`` so that the
stub and the real harness drive *the same* control flow. A drill that exercises
a different loop than the campaign proves nothing.

An OpenHands agent asks the user by emitting a MessageEvent with no tool call:
``run()`` then returns with execution status FINISHED and the last agent event
is a plain message rather than a FinishAction. That is the signal this file
turns into ``TurnResult(kind="question")``.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmio import PermanentError, TransientError, extract_reasoning, extract_usage, utcnow

TurnKind = str  # "question" | "finish" | "stuck" | "other"


@dataclass
class TurnResult:
    kind: TurnKind
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    response_id: str | None = None
    reasoning_text: str | None = None
    reasoning_opaque: str | None = None
    latency_s: float = 0.0
    n_events: int = 0
    agent_steps: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class AgentSession:
    """One episode's worth of agent. Subclasses must be independent of each other."""

    def send(self, text: str) -> None: raise NotImplementedError
    def run(self) -> TurnResult: raise NotImplementedError
    def read_solution(self) -> str | None: raise NotImplementedError
    def stats(self) -> dict[str, Any]: raise NotImplementedError
    def close(self) -> None: pass


# --------------------------------------------------------------------------
# OpenHands
# --------------------------------------------------------------------------

class OpenHandsSession(AgentSession):
    def __init__(self, *, model_cfg, workdir: Path, trace_dir: Path,
                 system_prompt: str | None, max_iterations: int,
                 budget_usd: float | None, keep_trace: bool = False):
        # Imported here so stub mode and the drills never need the SDK.
        from openhands.sdk import LLM, Agent, Conversation, Tool
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.terminal import TerminalTool

        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        # Event log and raw completions go OUTSIDE the workspace. The agent can
        # list and read anything in its workspace, and it does: with these
        # inside, an agent spent 23 of its 30 iterations reading its own
        # completion logs instead of solving the task, then hit the iteration
        # cap. The workspace holds the task and nothing else.
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        import config as _cfg
        agent_cfg = _cfg.AGENT
        self._step_chars = agent_cfg.step_text_chars
        self._keep_trace = keep_trace

        llm_kwargs: dict[str, Any] = dict(
            model=model_cfg.litellm_model,
            api_key=model_cfg.api_key() or "local",
            max_output_tokens=model_cfg.max_tokens,
            input_cost_per_token=model_cfg.input_cost_per_token,
            output_cost_per_token=model_cfg.output_cost_per_token,
            drop_params=True,
            usage_id=f"harness-{model_cfg.key}",
            native_tool_calling=model_cfg.native_tool_calling,
            # Written out rather than inherited: the SDK defaults are
            # temperature=None (provider's choice, so not reproducible),
            # seed=None, and log_completions=False.
            temperature=model_cfg.temperature,
            seed=model_cfg.seed,
            caching_prompt=agent_cfg.caching_prompt,
            log_completions=agent_cfg.log_completions or keep_trace,
            log_completions_folder=str(self.trace_dir / "completions"),
            max_message_chars=agent_cfg.max_message_chars,
            num_retries=agent_cfg.num_retries,
            retry_min_wait=agent_cfg.retry_min_wait,
            retry_max_wait=agent_cfg.retry_max_wait,
            timeout=agent_cfg.timeout,
        )
        if model_cfg.top_p is not None:
            llm_kwargs["top_p"] = model_cfg.top_p
        if model_cfg.base_url():
            llm_kwargs["base_url"] = model_cfg.base_url()
        # Written out explicitly per model in config.py; never inferred here.
        for k, v in model_cfg.extra.items():
            if k == "chat_template_kwargs":
                llm_kwargs["litellm_extra_body"] = {"chat_template_kwargs": v}
            else:
                llm_kwargs[k] = v

        self.llm = LLM(**llm_kwargs)
        agent_kwargs: dict[str, Any] = dict(
            llm=self.llm,
            tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
        )
        if agent_cfg.use_condenser:
            # Constructing an Agent by hand gives condenser=None (no
            # summarisation, unbounded history). The official preset uses
            # max_size=80, keep_first=4.
            from openhands.sdk.context.condenser import LLMSummarizingCondenser
            agent_kwargs["condenser"] = LLMSummarizingCondenser(
                llm=self.llm,
                max_size=agent_cfg.condenser_max_size,
                keep_first=agent_cfg.condenser_keep_first,
            )
        if system_prompt:
            # Tag arm only -- the protocol arm passes system_prompt=None and
            # carries SYSTEM_PROMPT in its first user message instead, so it
            # keeps both the OpenHands system prompt and the tools.
            # This is a FULL override (verified against SDK
            # 1.44.1: Agent exposes system_prompt, not system_prompt_suffix).
            # That is what the arm wants: no tools, no harness scaffold, so the
            # difference against the other arms isolates the scaffold from the
            # backbone.
            agent_kwargs["system_prompt"] = system_prompt
            agent_kwargs["tools"] = []
        self.agent = Agent(**agent_kwargs)

        conv_kwargs: dict[str, Any] = dict(
            agent=self.agent,
            workspace=str(self.workdir),
            max_iteration_per_run=max_iterations,
            stuck_detection=agent_cfg.stuck_detection,
            persistence_dir=str(self.trace_dir / "events"),
        )
        self.conv = Conversation(**conv_kwargs)
        if budget_usd:
            # Conversation is a factory and its __new__ does not take this;
            # LocalConversation stores it as a plain attribute (SDK 1.44.1), so
            # set it after construction. Verified: without it the default is
            # None, i.e. no cap at all.
            self.conv.max_budget_per_run = budget_usd
        self._seen = 0
        self._usage_seen = 0

    def send(self, text: str) -> None:
        self.conv.send_message(text)

    def run(self) -> TurnResult:
        from openhands.sdk.conversation.state import ConversationExecutionStatus as S
        from openhands.sdk.event import ActionEvent, MessageEvent
        from openhands.sdk.tool.builtins.finish import FinishAction

        t0 = time.time()
        self.conv.run()
        latency = time.time() - t0

        events = list(self.conv.state.events)
        new = events[self._seen:]
        self._seen = len(events)

        status = getattr(self.conv.state, "execution_status", None)
        last = None
        for e in reversed(events):
            if isinstance(e, ActionEvent):
                last = e
                break
            if isinstance(e, MessageEvent) and getattr(e, "source", None) == "agent":
                last = e
                break

        usage, reasoning_text, reasoning_opaque, finish_reason, resp_id = self._telemetry(new)
        steps = self._step_summaries(new)
        base = dict(usage=usage, reasoning_text=reasoning_text,
                    reasoning_opaque=reasoning_opaque, finish_reason=finish_reason,
                    response_id=resp_id, latency_s=latency, n_events=len(new),
                    agent_steps=steps,
                    raw={"n_total_events": len(events), "status": str(status)})

        if isinstance(last, ActionEvent) and isinstance(getattr(last, "action", None), FinishAction):
            sol = self.read_solution()
            base["raw"]["has_solution"] = bool(sol and sol.strip())
            return TurnResult(kind="finish", text=self._text_of(last), **base)
        if status is not None and status != S.FINISHED:
            # Ran out of iterations, out of budget, or stuck detection fired.
            return TurnResult(kind="stuck", text=self._text_of(last), **base)
        if isinstance(last, MessageEvent):
            return TurnResult(kind="question", text=self._text_of(last), **base)
        return TurnResult(kind="other", text=self._text_of(last), **base)

    @staticmethod
    def _text_of(event: Any) -> str:
        """Pull the agent's words out of any event.

        Verified against SDK 1.44.1: a MessageEvent carries llm_message.content,
        while an ActionEvent carries the tool arguments (FinishAction has a
        single `message` field) plus a `thought` list. The agent routinely puts
        a clarifying question in FinishAction.message, so failing to read it
        here would score a question as silence.
        """
        if event is None:
            return ""
        action = getattr(event, "action", None)
        if action is not None:
            parts = []
            m = getattr(action, "message", None)
            if isinstance(m, str) and m.strip():
                parts.append(m)
            if not parts:
                thought = getattr(event, "thought", None) or []
                for c in thought:
                    t = getattr(c, "text", None)
                    if t is None and isinstance(c, dict):
                        t = c.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            if parts:
                return "\n".join(parts)
        msg = getattr(event, "llm_message", None) or getattr(event, "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for c in content:
                t = getattr(c, "text", None)
                if t is None and isinstance(c, dict):
                    t = c.get("text")
                if isinstance(t, str):
                    parts.append(t)
            if parts:
                return "\n".join(parts)
        for attr in ("content", "text", "thought"):
            v = getattr(event, attr, None)
            if isinstance(v, str) and v:
                return v
        return repr(event)[:4000]

    def _token_usages(self) -> list[Any]:
        """Flat list of TokenUsage records accumulated so far.

        Verified against SDK 1.44.1: events carry only llm_response_id, so the
        numbers come from conversation_stats.get_combined_metrics(), which
        accumulates. Per-turn usage is the tail added since the last snapshot.
        """
        try:
            stats = self.conv.conversation_stats
            return list(stats.get_combined_metrics().token_usages or [])
        except Exception:
            return []

    def _telemetry(self, events: list[Any]) -> tuple[dict, str | None, str | None, str | None, str | None]:
        usage_total: dict[str, Any] = {}
        rtext = ropaque = freason = rid = None

        usages = self._token_usages()
        for tu in usages[self._usage_seen:]:
            for src, dst in (("prompt_tokens", "prompt_tokens"),
                             ("completion_tokens", "completion_tokens"),
                             ("reasoning_tokens", "reasoning_tokens"),
                             ("cache_read_tokens", "cached_tokens")):
                v = getattr(tu, src, None)
                if v is not None:
                    usage_total[dst] = (usage_total.get(dst) or 0) + v
            rid = getattr(tu, "response_id", None) or rid
        self._usage_seen = len(usages)
        if usage_total.get("prompt_tokens") is not None:
            usage_total["total_tokens"] = ((usage_total.get("prompt_tokens") or 0)
                                           + (usage_total.get("completion_tokens") or 0))

        # The reasoning trace is on the events themselves, in three possible
        # shapes. A null trace means the SDK did not expose one, never that the
        # model did not think -- reasoning_tokens above is the authority.
        for e in events:
            t = getattr(e, "reasoning_content", None)
            if isinstance(t, str) and t:
                rtext = t
            blocks = getattr(e, "thinking_blocks", None)
            if blocks:
                try:
                    parts = [getattr(b, "thinking", None) or (b.get("thinking") if isinstance(b, dict) else None)
                             for b in blocks]
                    joined = "\n".join(p for p in parts if p)
                    if joined:
                        rtext = joined
                except Exception:
                    pass
            item = getattr(e, "responses_reasoning_item", None)
            if item is not None:
                ropaque = repr(item)[:8000]
            th = getattr(e, "thought", None)
            if isinstance(th, str) and th and not rtext:
                rtext = th
        return usage_total, rtext, ropaque, freason, rid

    def _step_summaries(self, events: list[Any]) -> list[dict[str, Any]]:
        """What the agent did inside one run(), small enough to keep forever.

        The full completions dump is ~350 KB per episode and 90% of it is the
        same system prompt and history re-serialised on every step. This keeps
        the part that cannot be reconstructed: which tool ran, with what
        arguments, and what came back.
        """
        from openhands.sdk.event import ActionEvent, ObservationEvent

        n = self._step_chars
        out: list[dict[str, Any]] = []
        for e in events:
            if isinstance(e, ActionEvent):
                action = getattr(e, "action", None)
                try:
                    args = action.model_dump() if hasattr(action, "model_dump") else dict(
                        getattr(action, "__dict__", {}))
                except Exception:
                    args = {"repr": repr(action)[:n]}
                args = {k: (v[:n] if isinstance(v, str) else v) for k, v in args.items()}
                out.append({
                    "type": "action",
                    "tool": getattr(e, "tool_name", None),
                    "args": args,
                    "thought": self._joined_text(getattr(e, "thought", None))[:n],
                    "reasoning_content": (getattr(e, "reasoning_content", None) or "")[:n] or None,
                })
            elif isinstance(e, ObservationEvent):
                obs = getattr(e, "observation", None)
                text = ""
                for attr in ("text", "content", "output"):
                    v = getattr(obs, attr, None)
                    if isinstance(v, str) and v:
                        text = v
                        break
                if not text:
                    text = repr(obs)
                out.append({
                    "type": "observation",
                    "tool": getattr(e, "tool_name", None),
                    "text": text[:n],
                    "truncated": len(text) > n,
                })
        return out

    @staticmethod
    def _joined_text(seq: Any) -> str:
        if not seq:
            return ""
        parts = []
        for c in seq:
            t = getattr(c, "text", None)
            if t is None and isinstance(c, dict):
                t = c.get("text")
            if isinstance(t, str):
                parts.append(t)
        return "\n".join(parts)

    def read_solution(self) -> str | None:
        p = self.workdir / "solution.py"
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
        return None

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            m = self.conv.conversation_stats.get_combined_metrics()
            out["accumulated_cost"] = getattr(m, "accumulated_cost", None)
            out["model_name"] = getattr(m, "model_name", None)
            out["costs"] = json.loads(json.dumps(getattr(m, "costs", None), default=str))
            out["response_latencies"] = json.loads(
                json.dumps(getattr(m, "response_latencies", None), default=str))
            out["token_usages"] = json.loads(
                json.dumps(getattr(m, "token_usages", None), default=str))
        except Exception as exc:  # noqa: BLE001
            out["conversation_stats_error"] = f"{type(exc).__name__}: {exc}"
        return out

    def close(self) -> None:
        try:
            self.conv.close()
        except Exception:
            pass
        # The summary is already in the row; the raw dump is 350 KB of mostly
        # re-serialised history and there is not enough disk for 984 of them.
        if not self._keep_trace:
            shutil.rmtree(self.trace_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Stub
# --------------------------------------------------------------------------

class StubSession(AgentSession):
    """Canned agent. Zero API calls, full pipeline, and fault injection.

    Behaviour is deterministic per (task, arm, sample) so a resumed run
    reproduces the same episode. Faults are injected by env var, which is what
    makes the R9 drills possible without spending anything.

      HARNESS_STUB_FAIL=429:0.2   transient on 20% of turns
      HARNESS_STUB_FAIL=400:1.0   permanent on every turn (circuit breaker drill)
      HARNESS_STUB_SLEEP=0.05     per-turn latency
    """

    def __init__(self, *, task: dict[str, Any], arm: str, sample: int, workdir: Path,
                 max_questions: int):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.arm = arm
        self.max_questions = max_questions
        seed = f"{task.get('task_id')}|{arm}|{sample}"
        self.rng = random.Random(seed)
        self.qa = task.get("qa_pairs") or []
        # native arm asks much less often, which is the effect under study.
        p_ask = {"native": 0.25, "ask_permitted": 0.85, "tag": 0.80}.get(arm, 0.5)
        n_max = min(len(self.qa), max_questions)
        self.plan_asks = sum(1 for _ in range(n_max) if self.rng.random() < p_ask)
        self.asked = 0
        self.inbox: list[str] = []

    def send(self, text: str) -> None:
        self.inbox.append(text)

    def _inject(self) -> None:
        spec = os.environ.get("HARNESS_STUB_FAIL", "")
        if not spec:
            return
        code, _, rate = spec.partition(":")
        if self.rng.random() >= float(rate or 1.0):
            return
        exc: Any = Exception(f"injected {code}")
        exc.status_code = int(code)
        raise exc

    def run(self) -> TurnResult:
        time.sleep(float(os.environ.get("HARNESS_STUB_SLEEP", "0") or 0))
        self._inject()
        usage = {"prompt_tokens": 900 + 120 * self.asked, "completion_tokens": 180,
                 "total_tokens": 1080 + 120 * self.asked, "reasoning_tokens": 0,
                 "cached_tokens": 0}
        if self.asked < self.plan_asks and self.asked < len(self.qa):
            q = self.qa[self.asked]["question"]
            self.asked += 1
            # Half the questions go out through FinishAction rather than a plain
            # message, because that is what the real agent does (observed on
            # SDK 1.44.1). The episode loop must score both as questions.
            if self.rng.random() < 0.5:
                return TurnResult(kind="finish", text=q, usage=usage,
                                  finish_reason="stop", response_id=f"stub-fq{self.asked}",
                                  latency_s=0.01, n_events=1,
                                  agent_steps=[{"type": "action", "tool": "finish",
                                                "args": {"message": q[:400]}}],
                                  raw={"stub": True, "has_solution": False})
            return TurnResult(kind="question", text=q, usage=usage,
                              finish_reason="stop", response_id=f"stub-{self.asked}",
                              latency_s=0.01, n_events=1,
                              agent_steps=[{"type": "action", "tool": None,
                                            "args": {"message": q[:400]}}],
                              raw={"stub": True})
        code = "def main():\n    pass\nmain()\n"
        (self.workdir / "solution.py").write_text(code, encoding="utf-8")
        return TurnResult(kind="finish", text="Wrote solution.py", usage=usage,
                          finish_reason="stop", response_id="stub-finish",
                          latency_s=0.01, n_events=1,
                          agent_steps=[
                              {"type": "action", "tool": "file_editor",
                               "args": {"command": "create", "path": "solution.py"}},
                              {"type": "observation", "tool": "file_editor",
                               "text": "File created.", "truncated": False},
                          ],
                          raw={"stub": True, "has_solution": True})

    def read_solution(self) -> str | None:
        p = self.workdir / "solution.py"
        return p.read_text(encoding="utf-8") if p.exists() else None

    def stats(self) -> dict[str, Any]:
        return {"stub": True, "planned_asks": self.plan_asks, "asked": self.asked}


# --------------------------------------------------------------------------

_CODE_FENCE = re.compile(r"```(?:python)?[^\n]*\n(.*?)```", re.DOTALL)
_CODE_TAG = re.compile(r"\[CODE\](.*?)\[/CODE\]", re.DOTALL)
_QUESTION_TAG = re.compile(r"\[QUESTION\](.*?)\[/QUESTION\]", re.DOTALL)


def extract_code(text: str) -> tuple[str, str]:
    """Return ``(code, which_parser_fired)``. The parser name goes in the row:
    a tag hit and a fence fallback are not the same observation."""
    if not text:
        return "", "empty"
    m = _CODE_TAG.search(text)
    if m:
        return m.group(1).strip(), "tag"
    m = _CODE_FENCE.search(text)
    if m:
        return m.group(1).strip(), "fence"
    return text.strip(), "whole_text"


def extract_question(text: str) -> tuple[str, str]:
    m = _QUESTION_TAG.search(text)
    if m:
        return m.group(1).strip(), "tag"
    return (text or "").strip(), "whole_text"


def looks_like_code(text: str) -> bool:
    return "[CODE]" in (text or "")


def make_workdir(root: Path, model: str, arm: str, task_id: str, sample: int) -> tuple[Path, Path]:
    """Return ``(workspace, trace_dir)``.

    The workspace lives OUTSIDE the results tree entirely, under
    ``<results-root>/../workspaces/<run>/``. Putting it at
    ``results/harness/<run>/work/`` -- inside the very directory the runner
    writes to -- let the agent see the whole experiment: a Claude Code agent
    listed its cwd, found `results/harness/...`, and recreated that path
    structure inside its own workspace, nesting 22 directories deep. Same class
    of bug as the earlier completions/ leak: whatever the agent can list must
    contain nothing of the runner's.

    trace stays under the results tree, since the agent never sees it.
    """
    leaf = f"{model}/{arm}/{task_id}__s{sample}"
    work = root.parent.parent / "workspaces" / root.name / leaf
    trace = root / "trace" / leaf
    for d in (work, trace):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    return work, trace


# --------------------------------------------------------------------------
# Claude Code
# --------------------------------------------------------------------------

class ClaudeCodeSession(AgentSession):
    """Drive Claude Code headlessly, one protocol turn per `claude -p` call.

    Multi-turn works through `--resume <session_id>`: the first call returns a
    session id in its JSON, and every later call replays into that session, so
    the agent keeps its full history and its workspace.

    Three things are load-bearing and were each found the hard way:

    * The model name must be a real Claude id. Claude Code validates the model
      string client-side and rejects anything else with
      `[claude-code:unrecognized_model]`, so the LiteLLM proxy advertises
      Claude ids while routing to Qwen.
    * The proxy env vars must be stripped. These hosts export
      HTTP_PROXY=127.0.0.1:10808, and Node does not honour the `127.*` wildcard
      form of no_proxy, so localhost requests get sent to the proxy and come
      back as `API Error: 503`.
    * stdin must be closed, or the CLI waits on it before starting.
    """

    def __init__(self, *, model_cfg, workdir: Path, trace_dir: Path,
                 claude_bin: str, base_url: str, auth_token: str,
                 model: str, small_model: str, timeout_s: int):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.claude_bin = claude_bin
        self.timeout_s = timeout_s
        self.session_id: str | None = None
        self.pending: list[str] = []
        self._calls: list[dict[str, Any]] = []

        env = dict(os.environ)
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                    "ALL_PROXY", "all_proxy"):
            env.pop(var, None)
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["no_proxy"] = "127.0.0.1,localhost"
        env["ANTHROPIC_BASE_URL"] = base_url
        env["ANTHROPIC_AUTH_TOKEN"] = auth_token
        env["ANTHROPIC_MODEL"] = model
        env["ANTHROPIC_SMALL_FAST_MODEL"] = small_model
        # Keep the agent's own config/state out of the operator's home.
        env["CLAUDE_CONFIG_DIR"] = str(self.trace_dir / "claude_home")
        self.env = env

    def send(self, text: str) -> None:
        self.pending.append(text)

    def run(self) -> TurnResult:
        import subprocess

        prompt = "\n".join(self.pending)
        self.pending.clear()
        cmd = [self.claude_bin, "-p", "--output-format", "json",
               "--dangerously-skip-permissions"]
        if self.session_id:
            cmd += ["--resume", self.session_id]

        # The prompt goes on stdin, not as an argv positional. With --resume,
        # Claude Code rejects the positional form outright ("Input must be
        # provided either through stdin or as a prompt argument", rc=1), and
        # stdin works for both the first call and the resumed ones -- so use
        # the one form that is valid in both.
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(self.workdir), env=self.env,
                              input=prompt, capture_output=True,
                              text=True, timeout=self.timeout_s)
        latency = time.time() - t0

        raw = (proc.stdout or "").strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"claude returned non-JSON (rc={proc.returncode}): "
                f"{raw[:300] or (proc.stderr or '')[:300]}")

        self.session_id = payload.get("session_id") or self.session_id
        self._calls.append({k: payload.get(k) for k in
                            ("session_id", "num_turns", "duration_api_ms",
                             "total_cost_usd", "terminal_reason", "is_error")})

        if payload.get("is_error"):
            raise RuntimeError(f"claude error: {str(payload.get('result'))[:300]}")

        text = payload.get("result") or ""
        u = payload.get("usage") or {}
        usage = {
            "prompt_tokens": (u.get("input_tokens") or 0)
                             + (u.get("cache_read_input_tokens") or 0)
                             + (u.get("cache_creation_input_tokens") or 0),
            "completion_tokens": u.get("output_tokens"),
            "cached_tokens": u.get("cache_read_input_tokens"),
            "reasoning_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens"),
        }
        usage["total_tokens"] = (usage["prompt_tokens"] or 0) + (usage["completion_tokens"] or 0)

        base = dict(usage=usage, finish_reason=payload.get("stop_reason"),
                    response_id=payload.get("session_id"),
                    reasoning_text=None, reasoning_opaque=None,
                    latency_s=latency, n_events=payload.get("num_turns") or 0,
                    agent_steps=[],
                    raw={"terminal_reason": payload.get("terminal_reason"),
                         "num_turns": payload.get("num_turns"),
                         "has_solution": bool((self.read_solution() or "").strip())})

        # Claude Code always ends its turn; what it produced decides the kind.
        # Tagged code, or a written solution.py, is a delivery; anything else
        # is treated as a message and routed through the question path, the
        # same rule the OpenHands arms use.
        if looks_like_code(text) or base["raw"]["has_solution"]:
            return TurnResult(kind="finish", text=text, **base)
        return TurnResult(kind="question", text=text, **base)

    def read_solution(self) -> str | None:
        p = self.workdir / "solution.py"
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
        return None

    def stats(self) -> dict[str, Any]:
        return {"harness": "claude-code", "session_id": self.session_id,
                "calls": self._calls,
                "cost_usd_reported": sum((c.get("total_cost_usd") or 0)
                                         for c in self._calls)}

    def close(self) -> None:
        shutil.rmtree(self.trace_dir / "claude_home", ignore_errors=True)
