from evaluate_frozen_baselines_on_ccb_v3 import (
    extract_question,
    trajectory_questions,
)


def test_extract_question_accepts_strict_wrapper() -> None:
    assert extract_question("[QUESTION]\nWhich behavior?\n[/QUESTION]") == (
        "Which behavior?"
    )


def test_extract_question_rejects_code() -> None:
    assert extract_question("[CODE]\nprint(1)\n[/CODE]") is None


def test_trajectory_questions_preserves_order() -> None:
    row = {
        "response_contents": [
            "[QUESTION]\nFirst?\n[/QUESTION]",
            "[QUESTION]\nSecond?\n[/QUESTION]",
            "[CODE]\npass\n[/CODE]",
        ]
    }
    assert trajectory_questions(row) == ["First?", "Second?"]
