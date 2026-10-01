from __future__ import annotations

import pytest

from contra import Candidate, ExecutionResult, discover_questions, verify_behavior


def candidate(name: str) -> Candidate:
    return Candidate(name, f"Which behavior for {name}?", "Choose A.", "Choose B.")


def test_discovery_combines_both_generators_and_regenerates_completions() -> None:
    contextual_inputs = []
    independent_inputs = []
    completion_inputs = []

    def contextual(requirement, retained):
        contextual_inputs.append((requirement, retained))
        return [candidate("contextual")]

    def independent(requirement):
        independent_inputs.append(requirement)
        return [candidate("independent")]

    def complete(requirement, question):
        completion_inputs.append((requirement, question))
        return ("Completed requirement A.", "Completed requirement B.")

    result = discover_questions(
        "Print a value.",
        contextual=contextual,
        independent=independent,
        review_merge=lambda requirement, values: values,
        regenerate_completions=complete,
    )
    assert result.rounds == 3
    assert result.converged
    assert len(contextual_inputs) == len(independent_inputs) == 3
    assert contextual_inputs[0][1] == ()
    assert len(contextual_inputs[1][1]) == 2
    assert independent_inputs == ["Print a value."] * 3
    assert len(completion_inputs) == 2
    assert [value.candidate_id for value in result.candidates] == ["contextual", "independent"]
    assert all(value.hypothesis_a == "Completed requirement A." for value in result.candidates)


def test_discovery_requires_two_consecutive_stagnant_rounds() -> None:
    additions = iter([[candidate("first")], [], [candidate("second")], [], []])
    result = discover_questions(
        "Print a value.",
        contextual=lambda requirement, retained: next(additions),
        independent=lambda requirement: [],
        review_merge=lambda requirement, values: values,
        regenerate_completions=lambda requirement, question: ("A.", "B."),
    )
    assert result.rounds == 5
    assert result.converged
    assert len(result.candidates) == 2


def test_empty_discovery_converges_without_completion_calls() -> None:
    result = discover_questions(
        "Print a value.",
        contextual=lambda requirement, retained: [],
        independent=lambda requirement: [],
        review_merge=lambda requirement, values: values,
        regenerate_completions=lambda *args: pytest.fail("no retained question"),
    )
    assert result.rounds == 2
    assert result.converged
    assert result.candidates == ()


def test_discovery_round_cap_does_not_claim_convergence() -> None:
    result = discover_questions(
        "Print a value.",
        contextual=lambda requirement, retained: [candidate(str(len(retained)))],
        independent=lambda requirement: [],
        review_merge=lambda requirement, values: values,
        regenerate_completions=lambda requirement, question: ("A.", "B."),
        max_rounds=4,
    )
    assert result.rounds == 4
    assert len(result.candidates) == 4
    assert not result.converged


@pytest.mark.parametrize("max_rounds", [0, -1, True, 1.5])
def test_discovery_rejects_invalid_round_cap(max_rounds) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        discover_questions(
            "Print a value.",
            contextual=lambda *args: [],
            independent=lambda *args: [],
            review_merge=lambda *args: [],
            regenerate_completions=lambda *args: ("A.", "B."),
            max_rounds=max_rounds,
        )


@pytest.mark.parametrize(
    "review",
    [
        lambda requirement, values: [candidate("unknown")],
        lambda requirement, values: [values[0], values[0]],
        lambda requirement, values: [
            values[0],
            Candidate("duplicate", values[0].question, "A.", "B."),
        ],
    ],
)
def test_discovery_rejects_invalid_review_output(review) -> None:
    with pytest.raises(ValueError):
        discover_questions(
            "Print a value.",
            contextual=lambda requirement, retained: [candidate("known")],
            independent=lambda requirement: [candidate("duplicate")],
            review_merge=review,
            regenerate_completions=lambda requirement, question: ("A.", "B."),
        )


@pytest.mark.parametrize("pair", [("same", "same"), ("", "B"), ["A", "B"], ("A",)])
def test_discovery_rejects_invalid_regenerated_completions(pair) -> None:
    with pytest.raises(ValueError):
        discover_questions(
            "Print a value.",
            contextual=lambda requirement, retained: [candidate("one")],
            independent=lambda requirement: [],
            review_merge=lambda requirement, values: values,
            regenerate_completions=lambda requirement, question: pair,
            max_rounds=1,
        )


PROGRAMS_A = ("a1", "a2", "a3")
PROGRAMS_B = ("b1", "b2", "b3")
INPUTS = tuple(str(value) for value in range(7))


def test_behavior_requires_two_votes_and_keeps_all_execution_evidence() -> None:
    calls = []

    def execute(program, stdin):
        calls.append((program, stdin))
        if stdin != "6":
            return ExecutionResult("same\n")
        if program == "a3":
            return ExecutionResult("outlier\n")
        return ExecutionResult("A\n" if program.startswith("a") else "B\n")

    result = verify_behavior(
        PROGRAMS_A, PROGRAMS_B, INPUTS,
        execute=execute, is_valid_for_both=lambda value: True,
    )
    assert result.different
    assert result.witness == result.cases[6]
    assert result.witness.stdin == "6"
    assert len(result.cases) == 7
    assert len(calls) == 6 * 7
    assert result.cases[6].stable_a == "A\n"
    assert result.cases[6].stable_b == "B\n"


def test_behavior_without_within_group_agreement_is_not_a_witness() -> None:
    result = verify_behavior(
        PROGRAMS_A, PROGRAMS_B, INPUTS,
        execute=lambda program, value: ExecutionResult(program),
        is_valid_for_both=lambda value: True,
    )
    assert not result.different
    assert result.witness is None
    assert all(case.stable_a is None and case.stable_b is None for case in result.cases)


@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_failed_executions_never_vote(failure) -> None:
    def execute(program, value):
        if program.startswith("a"):
            return ExecutionResult("A", returncode=1 if failure == "error" else 0,
                                   timed_out=failure == "timeout")
        return ExecutionResult("B")

    result = verify_behavior(
        PROGRAMS_A, PROGRAMS_B, INPUTS,
        execute=execute, is_valid_for_both=lambda value: True,
    )
    assert not result.different
    assert all(case.stable_a is None and case.stable_b == "B" for case in result.cases)


def test_empty_stdout_is_a_defined_stable_output() -> None:
    result = verify_behavior(
        PROGRAMS_A, PROGRAMS_B, INPUTS,
        execute=lambda program, value: ExecutionResult("" if program.startswith("a") else "B"),
        is_valid_for_both=lambda value: True,
    )
    assert result.different
    assert result.witness.stable_a == ""


@pytest.mark.parametrize(
    "programs_a,programs_b,inputs",
    [
        (PROGRAMS_A[:2], PROGRAMS_B, INPUTS),
        (PROGRAMS_A, (*PROGRAMS_B, "b4"), INPUTS),
        (PROGRAMS_A, PROGRAMS_B, INPUTS[:6]),
        (PROGRAMS_A, PROGRAMS_B, ("repeated",) * 7),
    ],
)
def test_behavior_enforces_the_exact_program_and_input_thresholds(
    programs_a, programs_b, inputs,
) -> None:
    with pytest.raises(ValueError):
        verify_behavior(
            programs_a, programs_b, inputs,
            execute=lambda *args: pytest.fail("invalid inputs must not execute"),
            is_valid_for_both=lambda value: True,
        )


def test_behavior_rejects_inputs_invalid_under_either_completion() -> None:
    with pytest.raises(ValueError, match="both completions"):
        verify_behavior(
            PROGRAMS_A, PROGRAMS_B, INPUTS,
            execute=lambda *args: pytest.fail("invalid inputs must not execute"),
            is_valid_for_both=lambda value: value != "6",
        )
