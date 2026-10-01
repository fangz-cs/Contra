"""Question matching, identical in behaviour to clarifycodebench.interact.

The judge decides whether a question the agent asked matches one of the task's
annotated key questions, and returns the reply the environment sends back. The
prompt and the three-vote majority are copied from the protocol so that harness
rows are scored by exactly the same instrument as CONTRA and the baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from llmio import CallResult, extract_usage, utcnow, with_retries
from prompts import ALREADY_ASKED, NOT_RELEVANT

JUDGE_PROMPT = """\
Task Description:
Your task is to determine if "Statement 1" and "Statement 2" are trying to clarify the same ambiguity in a given context.

Evaluation Criteria:
- If both statements address the same core ambiguity, even if they use different wording, language, or sentence structure, they are a "match".
- If the statements focus on different aspects of the ambiguity or different potential ambiguities, they are "not a match".

Output Format:
- If they match, return only "yes".
- If they do not match, return only "no".

Now, please evaluate the following two statements based on the rules above:

[Statement 1]:
{key_question}

[Statement 2]:
{model_question}

Decision:
"""


@dataclass
class JudgeCall:
    key_question: str
    model_question: str
    votes: list[str]
    verdict: bool
    usage: list[dict[str, Any]] = field(default_factory=list)
    at: str = ""


@dataclass
class JudgeOutcome:
    hit: int                       # 1 fresh hit, 0 no hit / already asked
    reply: str
    matched: str | None
    calls: list[JudgeCall] = field(default_factory=list)


class Judge:
    """``chat`` is injected so the stub backend can drive this offline."""

    def __init__(self, chat: Callable[..., CallResult], model: str, consistency: int = 3,
                 max_tokens: int = 512, extra: dict[str, Any] | None = None):
        self.chat = chat
        self.model = model
        self.consistency = consistency
        self.max_tokens = max_tokens
        self.extra = extra or {}
        self.calls: list[JudgeCall] = []

    def _vote(self, key_q: str, model_q: str) -> JudgeCall:
        prompt = JUDGE_PROMPT.format(key_question=key_q, model_question=model_q)
        votes, usages = [], []

        for _ in range(self.consistency):
            def _do() -> CallResult:
                kw = dict(self.extra)
                if "chat_template_kwargs" in kw:
                    kw["extra_body"] = {"chat_template_kwargs": kw.pop("chat_template_kwargs")}
                return self.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    **kw,
                )

            res, _attempts = with_retries(_do)
            votes.append((res.text or "").strip().lower())
            usages.append(res.usage)

        blank = sum(1 for v in votes if not v)
        if blank:
            # An empty verdict is not a "no". Reading it as one turns a budget
            # or configuration fault into a silent stream of NOT_RELEVANT.
            raise RuntimeError(
                f"judge returned {blank}/{len(votes)} empty verdicts for model "
                f"{self.model!r} (max_tokens={self.max_tokens}). Refusing to "
                f"score them as 'no match'."
            )
        yes = sum(1 for v in votes if v.startswith("yes"))
        return JudgeCall(
            key_question=key_q,
            model_question=model_q,
            votes=votes,
            verdict=yes * 2 > self.consistency,
            usage=usages,
            at=utcnow(),
        )

    def answer(self, model_question: str, qa_pairs: list[dict[str, str]],
               hit_questions: set[str]) -> JudgeOutcome:
        """Judge one question against every key question, in annotation order."""
        calls: list[JudgeCall] = []
        for qa in qa_pairs:
            key_q = qa["question"]
            call = self._vote(key_q, model_question)
            calls.append(call)
            self.calls.append(call)
            if call.verdict:
                if key_q in hit_questions:
                    return JudgeOutcome(0, ALREADY_ASKED, None, calls)
                return JudgeOutcome(1, qa["answer"], key_q, calls)
        return JudgeOutcome(0, NOT_RELEVANT, None, calls)
