"""The paper's three-program, shared-input behavioral comparison."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ValueError("execution output must be text")
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise ValueError("returncode must be an integer")
        if not isinstance(self.timed_out, bool):
            raise ValueError("timed_out must be a boolean")


@dataclass(frozen=True)
class CaseEvidence:
    stdin: str
    executions_a: tuple[ExecutionResult, ...]
    executions_b: tuple[ExecutionResult, ...]
    stable_a: str | None
    stable_b: str | None

    @property
    def separates(self) -> bool:
        return (
            self.stable_a is not None
            and self.stable_b is not None
            and self.stable_a != self.stable_b
        )


@dataclass(frozen=True)
class BehavioralResult:
    cases: tuple[CaseEvidence, ...]

    @property
    def different(self) -> bool:
        return any(case.separates for case in self.cases)

    @property
    def witness(self) -> CaseEvidence | None:
        return next((case for case in self.cases if case.separates), None)


def _stable_output(executions: tuple[ExecutionResult, ...]) -> str | None:
    votes = Counter(
        result.stdout
        for result in executions
        if result.returncode == 0 and not result.timed_out
    )
    return next((output for output, count in votes.items() if count >= 2), None)


def verify_behavior(
    programs_a: Sequence[str],
    programs_b: Sequence[str],
    inputs: Sequence[str],
    *,
    execute: Callable[[str, str], ExecutionResult],
    is_valid_for_both: Callable[[str], bool],
) -> BehavioralResult:
    """Compare three independently generated programs per completion.

    At least seven distinct inputs must be valid under both completions.
    Only successful executions vote, and output text is compared exactly.
    Every program runs on every input so the result keeps all evidence.
    A failed comparison does not establish behavioral equivalence.
    The caller must provide an isolated executor for model-generated code.
    """
    group_a, group_b, shared_inputs = tuple(programs_a), tuple(programs_b), tuple(inputs)
    if len(group_a) != 3 or len(group_b) != 3:
        raise ValueError("behavioral verification requires three programs per completion")
    if any(not isinstance(program, str) or not program.strip() for program in (*group_a, *group_b)):
        raise ValueError("programs must be non-empty strings")
    if any(not isinstance(value, str) for value in shared_inputs):
        raise ValueError("shared inputs must be strings")
    if len(set(shared_inputs)) < 7:
        raise ValueError("behavioral verification requires at least seven distinct inputs")
    if any(is_valid_for_both(value) is not True for value in shared_inputs):
        raise ValueError("an input is not valid under both completions")

    cases: list[CaseEvidence] = []
    for stdin in shared_inputs:
        executions_a = tuple(execute(program, stdin) for program in group_a)
        executions_b = tuple(execute(program, stdin) for program in group_b)
        if any(
            not isinstance(result, ExecutionResult)
            for result in (*executions_a, *executions_b)
        ):
            raise ValueError("executor must return ExecutionResult objects")
        cases.append(
            CaseEvidence(
                stdin,
                executions_a,
                executions_b,
                _stable_output(executions_a),
                _stable_output(executions_b),
            )
        )
    return BehavioralResult(tuple(cases))
