"""One episode = one cell. The loop is shared by every backend on purpose.

The row this produces is meant to be self-sufficient: the test is whether a new
question about this rollout can be answered in six months without calling the
API again. So every turn keeps its raw text, which parser fired, its usage, its
timing, and every environment reply verbatim.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import backends
from backends import AgentSession, TurnResult
from judge import Judge
from llmio import PermanentError, TransientError, utcnow, with_retries
from prompts import FOLLOWUP, NO_MORE_ANSWERS, NUDGE, first_message, tag_system_prompt


@dataclass
class Turn:
    index: int
    kind: str                       # question | finish | stuck | other
    raw_text: str
    parsed_type: str                # question | code | none
    parser: str                     # tag | fence | whole_text | empty
    parsed_value: str
    env_reply: str | None           # verbatim, exactly what was sent back
    judged_hit: int | None
    matched_key_question: str | None
    counted_against_budget: bool
    usage: dict[str, Any]
    finish_reason: str | None
    response_id: str | None
    reasoning_text: str | None
    reasoning_opaque: str | None
    latency_s: float
    n_events: int
    agent_steps: list[dict[str, Any]]
    started_at: str
    ended_at: str
    judge_calls: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)


def run_episode(*, session: AgentSession, task: dict[str, Any], arm: str,
                judge: Judge, max_questions: int, max_turns: int) -> dict[str, Any]:
    """Drive one agent to a terminal state. Raises on a permanent API failure."""
    qa_pairs = task.get("qa_pairs") or []
    hit_questions: set[str] = set()
    hit_dict: dict[str, int] = {}
    asked: list[str] = []
    turns: list[Turn] = []
    messages: list[dict[str, str]] = []

    opening = first_message(arm, task["modified_content"])
    session.send(opening)
    messages.append({"role": "user", "content": opening})

    terminal = "max_turns"
    for i in range(max_turns):
        t_start = utcnow()
        res, attempts = with_retries(session.run)
        t_end = utcnow()
        messages.append({"role": "assistant", "content": res.text})

        turn = Turn(
            index=i, kind=res.kind, raw_text=res.text, parsed_type="none",
            parser="empty", parsed_value="", env_reply=None, judged_hit=None,
            matched_key_question=None, counted_against_budget=False,
            usage=res.usage, finish_reason=res.finish_reason,
            response_id=res.response_id, reasoning_text=res.reasoning_text,
            reasoning_opaque=res.reasoning_opaque, latency_s=res.latency_s,
            n_events=res.n_events, agent_steps=res.agent_steps,
            started_at=t_start, ended_at=t_end,
            attempts=[a.__dict__ for a in attempts],
        )

        if res.kind == "stuck":
            turn.parsed_type = "code"
            terminal = res.kind
            turns.append(turn)
            break

        if res.kind == "finish":
            # An agent that calls finish WITHOUT writing a program has not
            # delivered anything; measured on SDK 1.44.1, it routinely uses
            # FinishAction.message to ask its clarifying question instead of
            # sending a plain message. Treating that as "finished, asked
            # nothing" would score a question as silence -- which is exactly
            # the quantity this study measures. So a finish with no program is
            # routed through the question path like any other message.
            if res.raw.get("has_solution"):
                turn.parsed_type = "code"
                terminal = "finish"
                turns.append(turn)
                break
            if arm in ("tag", "protocol") and backends.looks_like_code(res.text):
                # finish without a file, but the message carries tagged code:
                # the protocol's own contract, so accept it as delivery.
                code, parser = backends.extract_code(res.text)
                turn.parsed_type, turn.parser, turn.parsed_value = "code", parser, code
                terminal = "finish_tagged_code"
                turns.append(turn)
                break
            turn.kind = "finish_no_code"

        # A message with no tool call. In the tag and protocol arms it may
        # carry [CODE] instead: those arms are given ClarifyCodeBench's output
        # contract, so the agent is entitled to answer in tags.
        if arm in ("tag", "protocol") and backends.looks_like_code(res.text):
            code, parser = backends.extract_code(res.text)
            turn.parsed_type, turn.parser, turn.parsed_value = "code", parser, code
            terminal = "code_in_message"
            turns.append(turn)
            break

        question, parser = backends.extract_question(res.text)
        turn.parser = parser

        if not question.strip():
            # Nothing to judge. Nudge without spending a question.
            turn.parsed_type = "none"
            turn.env_reply = NUDGE
            session.send(NUDGE)
            messages.append({"role": "user", "content": NUDGE})
            turns.append(turn)
            continue

        turn.parsed_type = "question"
        turn.parsed_value = question

        if len(asked) >= max_questions:
            turn.env_reply = NO_MORE_ANSWERS
            session.send(NO_MORE_ANSWERS)
            messages.append({"role": "user", "content": NO_MORE_ANSWERS})
            turns.append(turn)
            continue

        outcome = judge.answer(question, qa_pairs, hit_questions)
        hit_dict[str(len(asked))] = outcome.hit
        asked.append(question)
        turn.counted_against_budget = True
        turn.judged_hit = outcome.hit
        turn.matched_key_question = outcome.matched
        turn.judge_calls = [
            {"key_question": c.key_question, "votes": c.votes,
             "verdict": c.verdict, "usage": c.usage, "at": c.at}
            for c in outcome.calls
        ]
        if outcome.matched:
            hit_questions.add(outcome.matched)

        reply = outcome.reply + FOLLOWUP
        turn.env_reply = reply
        session.send(reply)
        messages.append({"role": "user", "content": reply})
        turns.append(turn)

    # Final program: the file is authoritative, the finish message is fallback.
    solution = session.read_solution()
    if solution is not None and solution.strip():
        final_code, code_source = solution, "solution.py"
    else:
        last_text = turns[-1].raw_text if turns else ""
        final_code, parser = backends.extract_code(last_text)
        code_source = f"message:{parser}"

    return {
        "turns": [t.__dict__ for t in turns],
        "messages": messages,
        "n_turns": len(turns),
        "terminal": terminal,
        "asked_questions": asked,
        "n_asked": len(asked),
        "hit_dict": hit_dict,
        "hit_questions": sorted(hit_questions),
        "final_code": final_code,
        "final_code_source": code_source,
        "session_stats": session.stats(),
    }


def build_row(*, cell_key: str, run_id: str, git_commit: str, stub: bool,
              model_key: str, litellm_model: str, arm: str, sample: int,
              task: dict[str, Any], request_kwargs: dict[str, Any],
              episode: dict[str, Any] | None, status: str,
              started_at: str, ended_at: str, attempts: list[dict[str, Any]],
              error: dict[str, Any] | None, schema_version: int,
              config_digest: str) -> dict[str, Any]:
    """A complete, self-sufficient record of one unit of work."""
    row: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "git_commit": git_commit,
        "stub": stub,                       # checked on every read path
        "cell_key": cell_key,
        "config_digest": config_digest,
        # identity
        "model": model_key,
        "litellm_model": litellm_model,
        "arm": arm,
        "task_id": task.get("task_id"),
        "id": task.get("question_id"),
        "sample": sample,
        "key_question_num": len(task.get("qa_pairs") or []),
        "prompt": task.get("modified_content"),
        "qa_pairs": task.get("qa_pairs"),
        # request
        "request_kwargs": request_kwargs,
        # status / timing / retries
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "n_retries": len(attempts),
        "attempts": attempts,
        "error": error,
    }
    if episode is not None:
        row.update(episode)
        row["usage_total"] = _sum_usage(episode.get("turns") or [])
    return row


def _sum_usage(turns: list[dict[str, Any]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for t in turns:
        for k, v in (t.get("usage") or {}).items():
            if isinstance(v, int):
                total[k] = total.get(k, 0) + v
    for t in turns:
        for call in t.get("judge_calls") or []:
            for u in call.get("usage") or []:
                for k, v in (u or {}).items():
                    if isinstance(v, int):
                        total["judge_" + k] = total.get("judge_" + k, 0) + v
    return total
