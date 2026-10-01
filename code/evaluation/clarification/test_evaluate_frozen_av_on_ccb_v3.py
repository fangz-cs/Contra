#!/usr/bin/env python3
"""Unit tests for frozen AV evaluation on CCB V3."""
from __future__ import annotations

import evaluate_frozen_av_on_ccb_v3 as target


def test_parse_response_reorders_and_validates() -> None:
    text = (
        '{"candidate_matches":['
        '{"candidate_id":"c2","gold_ids":[]},'
        '{"candidate_id":"c1","gold_ids":["g2","g1"]}]}'
    )
    assert target.parse_response(
        text,
        candidate_ids=["c1", "c2"],
        gold_ids=["g1", "g2"],
    ) == [
        {"candidate_id": "c1", "gold_ids": ["g1", "g2"]},
        {"candidate_id": "c2", "gold_ids": []},
    ]


def test_swapped_or_missing_ids_rejected() -> None:
    text = (
        '{"candidate_matches":['
        '{"candidate_id":"c1","gold_ids":["g1"]},'
        '{"candidate_id":"c1","gold_ids":[]}]}'
    )
    try:
        target.parse_response(
            text,
            candidate_ids=["c1", "c2"],
            gold_ids=["g1"],
        )
    except target.EvaluationError:
        pass
    else:
        raise AssertionError("duplicate candidate ID was accepted")


def test_fleiss_kappa_perfect_agreement() -> None:
    assert target.fleiss_kappa([0, 0, 3, 3], 3) == 1.0


def test_bootstrap_delta_is_deterministic() -> None:
    rows = [
        {
            "question_id": "q1",
            "stage_hits": {"left": True, "right": False},
        },
        {
            "question_id": "q2",
            "stage_hits": {"left": True, "right": True},
        },
    ]
    result = target.bootstrap_delta(
        gold_rows=rows,
        left="left",
        right="right",
        samples=100,
        seed=7,
    )
    assert result["point_delta"] == -0.5
    assert result["cluster_count"] == 2


def test_batches_and_public_ids_are_short() -> None:
    unit = {
        "item_id": "v_task_1",
        "requirement": "S",
        "gold": [
            {"gold_id": "v_task_1::gold_01", "question": "G?", "answer": "A"}
        ],
        "candidates": [
            {"candidate_id": f"v_task_1::candidate_{i}", "question": f"Q{i}?"}
            for i in range(5)
        ],
    }
    batches = target.build_batches([unit], 3)
    assert [len(batch["candidates"]) for batch in batches] == [3, 2]
    payload, candidate_map, gold_map = target.public_unit(batches[0], 0)
    assert set(candidate_map) == {"c001", "c002", "c003"}
    assert gold_map == {"g01": "v_task_1::gold_01"}
    assert {
        row["candidate_id"] for row in payload["candidate_questions"]
    } == {"c001", "c002", "c003"}
