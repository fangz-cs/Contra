"""Discovery with contextual generation, independent generation, and merging."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    question: str
    hypothesis_a: str
    hypothesis_b: str

    def __post_init__(self) -> None:
        for value in (
            self.candidate_id,
            self.question,
            self.hypothesis_a,
            self.hypothesis_b,
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("candidate fields must be non-empty strings")
        if self.hypothesis_a.strip() == self.hypothesis_b.strip():
            raise ValueError("candidate hypotheses must differ")


@dataclass(frozen=True)
class DiscoveryResult:
    candidates: tuple[Candidate, ...]
    rounds: int
    converged: bool


def _index(candidates: Sequence[Candidate]) -> dict[str, Candidate]:
    indexed: dict[str, Candidate] = {}
    for candidate in candidates:
        if not isinstance(candidate, Candidate):
            raise ValueError("discovery callbacks must return Candidate objects")
        previous = indexed.get(candidate.candidate_id)
        if previous is not None and previous != candidate:
            raise ValueError(f"conflicting candidate ID: {candidate.candidate_id}")
        indexed[candidate.candidate_id] = candidate
    return indexed


def discover_questions(
    requirement: str,
    *,
    contextual: Callable[[str, tuple[Candidate, ...]], Sequence[Candidate]],
    independent: Callable[[str], Sequence[Candidate]],
    review_merge: Callable[[str, tuple[Candidate, ...]], Sequence[Candidate]],
    regenerate_completions: Callable[[str, str], tuple[str, str]],
    max_rounds: int = 10,
) -> DiscoveryResult:
    """Discover candidates until two consecutive rounds add no retained ID.

    The reviewer keeps a supplied ID for each retained or merged question.
    Completions are regenerated after discovery, without provisional answers.
    Callbacks own semantic judgments and must not read benchmark annotations.
    This function has no provider access, persistence, or batch execution.
    """
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("requirement must be a non-empty string")
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer")

    retained: tuple[Candidate, ...] = ()
    stagnant_rounds = 0
    converged = False
    for round_number in range(1, max_rounds + 1):
        combined = _index(
            (
                *retained,
                *contextual(requirement, retained),
                *independent(requirement),
            )
        )
        reviewed = tuple(review_merge(requirement, tuple(combined.values())))
        reviewed_index = _index(reviewed)
        if len(reviewed_index) != len(reviewed):
            raise ValueError("reviewer returned duplicate candidate IDs")
        if not reviewed_index.keys() <= combined.keys():
            raise ValueError("reviewer invented an unknown candidate ID")
        questions = [" ".join(item.question.split()).casefold() for item in reviewed]
        if len(set(questions)) != len(questions):
            raise ValueError("reviewer retained duplicate questions")

        added = reviewed_index.keys() - {item.candidate_id for item in retained}
        stagnant_rounds = 0 if added else stagnant_rounds + 1
        retained = reviewed
        if stagnant_rounds == 2:
            converged = True
            break

    completed: list[Candidate] = []
    for candidate in retained:
        pair = regenerate_completions(requirement, candidate.question)
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("completion callback must return a pair of strings")
        completed.append(
            replace(candidate, hypothesis_a=pair[0], hypothesis_b=pair[1])
        )
    return DiscoveryResult(tuple(completed), round_number, converged)
