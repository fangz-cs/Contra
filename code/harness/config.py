"""Explicit per-model configuration.

Every request kwarg is written out per model, by hand. Nothing is inferred from
the model id at runtime: picking kwargs by substring-matching a model name is
how a campaign ends up sending a rejected parameter on every call and every
retry (see R4).

``max_tokens`` values marked CALIBRATE are placeholders until ``preflight``
has measured them on the actual endpoint. Preflight refuses to pass a model
whose budget is still uncalibrated in a real run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    key: str                      # short name used in cell keys and filenames
    litellm_model: str            # what LiteLLM / OpenHands is handed
    base_url_env: str             # env var holding the endpoint
    api_key_env: str              # env var holding the key
    max_tokens: int
    calibrated: bool              # False => max_tokens is a placeholder
    # Cost, USD per token. 0.0 for a self-hosted model.
    input_cost_per_token: float
    output_cost_per_token: float
    # Extra kwargs, written out explicitly. Never defaulted by this file.
    extra: dict[str, Any] = field(default_factory=dict)
    # OpenHands only ever works with native tool calling; recorded so the row
    # says which transport produced it.
    native_tool_calling: bool = True
    # Sampling. The SDK leaves this None, i.e. "whatever the provider
    # defaults to", which is not reproducible. The value is not a free choice:
    # it must match what CONTRA and the baselines were run at, or the arms are
    # not comparable. From the shipped manifests --
    #   results/contra/{gemini-3.1-pro-preview,grok-4.5}: temperature 0.0
    #   results/baselines/*:                              temperature 0.0
    #   results/contra/{gpt-5-mini,gpt-5.5}:              temperature null
    # -- every model that accepts the parameter ran at 0.0, and null appears
    # only where the model rejects it. So: 0.0 where accepted, None where not.
    temperature: float | None = 0.0
    top_p: float | None = None
    seed: int | None = 42

    def base_url(self) -> str | None:
        return os.environ.get(self.base_url_env) or None

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


MODELS: dict[str, ModelConfig] = {
    # Self-hosted on the A100 pool. Requires the replica to be started with
    # --enable-auto-tool-choice --tool-call-parser qwen3_coder, otherwise every
    # request comes back 400 (a permanent failure, never retried).
    "qwen3.8-27b": ModelConfig(
        key="qwen3.8-27b",
        litellm_model="openai/Qwen/Qwen3.8-27B",
        base_url_env="CONTRA_QWEN_BASE_URL",
        api_key_env="CONTRA_QWEN_API_KEY",
        max_tokens=8192,          # CALIBRATE (see probe_budget.py results below)
        calibrated=False,
        input_cost_per_token=0.0,
        output_cost_per_token=0.0,
        # The deployment runs --reasoning-parser qwen3, so thinking is on by
        # default. It is turned off here, and that is a measurement decision,
        # not a convenience: probe_budget.py against this endpoint on
        # 2026-09-04 showed that with thinking ON, reasoning tokens are billed
        # to the SAME max_tokens budget as the answer --
        #
        #   thinking=ON  max_tokens=200  -> completion=200, reasoning=187,
        #                                   answer=27 bytes, finish=length
        #   thinking=ON  max_tokens=4000 -> completion=197, reasoning=120,
        #                                   answer=132 bytes, finish=stop
        #
        # so a budget set for the answer alone returns near-empty turns on hard
        # prompts. In a study that codes WHETHER THE AGENT ASKED, an empty turn
        # is scored as "it said nothing" -- the exact quantity being measured.
        # If thinking is turned back on, raise max_tokens to well above the
        # measured reasoning+answer p99 first, and re-run the drills.
        #
        # Also measured: reasoning_content came back EMPTY while
        # reasoning_tokens was non-zero. The trace is not exposed by this
        # endpoint; the token count is the only evidence that it thought.
        #
        # No reasoning_effort here: this is a vLLM endpoint that does not take
        # it, and thinking is controlled by chat_template_kwargs instead.
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "gpt-5-mini": ModelConfig(
        key="gpt-5-mini",
        litellm_model="openai/gpt-5-mini",
        base_url_env="CONTRA_OPENAI_BASE_URL",
        api_key_env="CONTRA_OPENAI_API_KEY",
        max_tokens=16384,         # CALIBRATE
        calibrated=False,
        input_cost_per_token=0.25e-6,
        output_cost_per_token=2.00e-6,
        # None, not 0.0: this model rejects the parameter, and the CONTRA
        # manifest for it records temperature null for the same reason.
        temperature=None,
        seed=None,
        extra={"reasoning_effort": "low"},
    ),
    "gemini-3.1-pro": ModelConfig(
        key="gemini-3.1-pro",
        litellm_model="gemini/gemini-3.1-pro-preview",
        base_url_env="CONTRA_GEMINI_BASE_URL",
        api_key_env="CONTRA_GEMINI_API_KEY",
        # Gemini bills thinking to the completion budget on some gateways; the
        # probe in preflight is what decides this, not this comment.
        max_tokens=32768,         # CALIBRATE
        calibrated=False,
        input_cost_per_token=2.00e-6,
        output_cost_per_token=12.00e-6,
        # NOT empty: the SDK's default reasoning_effort is "high". Leaving this
        # out would silently run one backbone at high while the paper reports
        # low for every other arm.
        extra={"reasoning_effort": "low"},
    ),
}


@dataclass(frozen=True)
class AgentConfig:
    """Agent- and conversation-level knobs.

    Each of these has an SDK default that is wrong for this experiment in at
    least one way, so all of them are written out here rather than inherited.
    """
    # Conversation caps. SDK defaults: 500 iterations, no budget at all.
    #
    # Measured over 11 episodes on ClarifyCodeBench (Qwen3.8-27B, both arms),
    # after the workspace-contamination fix: steps per run() were
    #     1, 1, 1, 2, 2, 4, 5, 5, 5, 10, 30
    # The distribution is bimodal. Everything that finished did so in <= 10
    # steps. The single 30 was a degenerate loop -- the agent wrote
    # explore3..explore12.py while re-running explore2.py every time, so steps
    # 11-29 produced nothing. stuck_detection did NOT catch it; the cap is the
    # only backstop.
    #
    # 15 leaves 50% headroom over the worst productive episode and halves what
    # a degenerate loop costs. A capped episode burned 436k prompt tokens
    # against 30k for a normal one (14.5x), so the tail is where the GPU hours
    # actually go.
    max_iteration_per_run: int = 15
    max_budget_per_task_usd: float | None = 1.50
    # SDK default True. It ends an episode the agent is looping in, which is
    # what we want, but it means "stuck" is a distinct terminal state that has
    # to be reported separately from "finished".
    stuck_detection: bool = True

    # History condensation. The official preset uses max_size=80, keep_first=4;
    # constructing an Agent by hand (as backends.py does) gives condenser=None,
    # i.e. NO summarisation and an unbounded history.
    #
    # Trade-off worth deciding deliberately: with the condenser ON, a long
    # episode's early turns get summarised, so what the agent "remembers about
    # having already asked" is a summary rather than the transcript. With it
    # OFF, context grows until the 81920-token window is hit and the request
    # fails outright. Measured: ~7.8k tokens on the first call, ~3k added per
    # agent iteration, so a 5-question episode lands in the 30-45k range and
    # fits -- but not with much room.
    use_condenser: bool = False
    condenser_max_size: int = 80
    condenser_keep_first: int = 4

    # Observation truncation (SDK default 30000 chars). Long terminal output is
    # cut to this before it enters the history.
    max_message_chars: int = 30000
    # SDK default True. Harmless on a local endpoint, real money elsewhere.
    caching_prompt: bool = True
    # SDK default False, and it STAYS false for the campaign.
    #
    # Measured: a full trace (completions + events) is ~350 KB per episode, so
    # 984 cells would need ~344 GB. The A100 has 23 GB free. And ~90% of that
    # volume is redundant -- every step re-serialises the same 8k system prompt
    # and the whole history.
    #
    # What is actually irreplaceable is what the agent DID: the tool calls, the
    # observations, and the tokens each step cost. That is extracted into the
    # row itself as `agent_steps` (see backends._step_summaries), which costs
    # ~6 KB per episode instead of 350 KB.
    #
    # keep_full_trace_first_n cells still get the untouched trace, so the
    # summary can be audited against the raw article it came from.
    log_completions: bool = False
    keep_full_trace_first_n: int = 20
    # How much of each command / observation to keep in the summary.
    step_text_chars: int = 400
    # SDK defaults: 5 / 8s / 64s / 8.0. These sit UNDER our own retry layer.
    num_retries: int = 5
    retry_min_wait: int = 8
    retry_max_wait: int = 64
    # 300 (the SDK default) is too tight for this deployment and produces
    # timeouts that are pure waste. Measured on the A6000: total generation
    # throughput ~90 tok/s, shared by `workers` concurrent agents, so at
    # workers=4 a single request gets ~22 tok/s. An 8192-token completion then
    # needs ~370s -- and that is before the 40k-token prefill. Every such
    # request timed out at 300s and was retried up to 3 times, costing ~15
    # minutes per episode and producing nothing. The requests were not failing;
    # they were being cut off while still working.
    timeout: int = 900


AGENT = AgentConfig()


@dataclass(frozen=True)
class ClaudeCodeConfig:
    """Claude Code driven headlessly against a local Anthropic-protocol proxy.

    The proxy (LiteLLM) speaks /v1/messages and forwards to the Qwen vLLM, so
    the harness is Claude Code but the backbone is the same Qwen3.8-27B every
    other arm uses. Model names must be real Claude ids: Claude Code validates
    the string client-side and rejects anything else.
    """
    claude_bin_env: str = "CONTRA_CLAUDE_BIN"
    claude_bin_default: str = "claude"
    base_url_env: str = "CONTRA_CC_BASE_URL"
    base_url_default: str = "http://127.0.0.1:4100"
    auth_token_env: str = "CONTRA_CC_AUTH_TOKEN"
    auth_token_default: str = "sk-1234"
    model: str = "claude-sonnet-4-5-20250929"
    small_model: str = "claude-3-5-haiku-20241022"
    timeout_s: int = 900

    def claude_bin(self) -> str:
        return os.environ.get(self.claude_bin_env, self.claude_bin_default)

    def base_url(self) -> str:
        return os.environ.get(self.base_url_env, self.base_url_default)

    def auth_token(self) -> str:
        return os.environ.get(self.auth_token_env, self.auth_token_default)


CLAUDE_CODE = ClaudeCodeConfig()


@dataclass(frozen=True)
class JudgeConfig:
    # The local Qwen, by decision, because there is no OpenAI quota. This is a
    # deliberate substitution, not a default that drifted in, and it carries an
    # obligation: EVERY arm must be rescored with it. A harness scored by Qwen
    # and a CONTRA scored by gpt-4o cannot be compared -- use rescore_all.py.
    #
    # Calibrated against 300 stratified pairs of gpt-4o verdicts reconstructed
    # from the shipped CONTRA records, with thinking off and the protocol's
    # 3-vote majority (calibrate_judge.py):
    #     agreement 0.727   precision 0.972   recall 0.467   kappa 0.453
    # It is systematically stricter than gpt-4o -- 80 false negatives against 2
    # false positives -- so absolute hit rates come out LOWER than the numbers
    # in the shipped manifests. That is acceptable only because every arm takes
    # the same hit; the paper must report the judge change and these figures.
    model: str = os.environ.get("CONTRA_JUDGE_MODEL", "Qwen/Qwen3.8-27B")
    base_url_env: str = "CONTRA_JUDGE_BASE_URL"
    api_key_env: str = "CONTRA_JUDGE_API_KEY"
    consistency: int = 3          # majority over 3 votes, as in the protocol
    # The verdict is one word, but 16 tokens is only safe on a model that does
    # not think. On a reasoning model the thinking consumes the budget and the
    # verdict comes back as an EMPTY STRING -- which is then read as "no match",
    # so every question scores NOT_RELEVANT and every hit rate is silently zero.
    # Observed on Qwen3.8-27B at max_tokens=16. Preflight checks for it.
    max_tokens: int = 512
    input_cost_per_token: float = 2.50e-6
    output_cost_per_token: float = 10.00e-6
    # Provider-specific kwargs, same idea as ModelConfig.extra. Needed when the
    # judge endpoint is a local vLLM whose thinking has to be turned off.
    extra: dict[str, Any] = field(default_factory=dict)

    def base_url(self) -> str | None:
        return os.environ.get(self.base_url_env) or None

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


def _judge_extra() -> dict[str, Any]:
    """Turn thinking off when the judge is served by the local vLLM."""
    if "qwen" in JudgeConfig.model.lower():
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


JUDGE = JudgeConfig(extra=_judge_extra())

# max_questions matches the CONTRA protocol's 5-question budget.
MAX_QUESTIONS = 5
MAX_ITERATIONS_PER_RUN = AGENT.max_iteration_per_run
MAX_BUDGET_PER_TASK_USD = AGENT.max_budget_per_task_usd


def resolved_config() -> dict[str, Any]:
    """The config as it will be recorded in the manifest."""
    return {
        "models": {k: asdict(v) for k, v in MODELS.items()},
        "judge": asdict(JUDGE),
        "agent": asdict(AGENT),
        "claude_code": asdict(CLAUDE_CODE),
        "max_questions": MAX_QUESTIONS,
    }
