from __future__ import annotations

import json

import pytest

import run_agent2_contrastive_resolution as agent2
import run_av_decision_resolution as base


@pytest.fixture
def chunk():
    candidate = base.Candidate(
        candidate_id="variant::format",
        group_id="variant",
        requirement="Print all squares on one line, separated by spaces.",
        question="Use one line or multiple lines?",
        hypothesis_a="Use one line.",
        hypothesis_b="Use multiple lines.",
    )
    return base.build_chunks(
        [candidate], stage="decision_relevance",
        max_candidates=32, max_prompt_chars=60000,
    )[0]


def response(chunk, resolution, spans):
    return json.dumps({
        "resolutions": [{
            "candidate_id": chunk.items[0].local_id,
            "resolution": resolution,
            "evidence_spans": spans,
        }],
    })


def test_contrastive_resolution_accepts_verbatim_evidence(chunk) -> None:
    parsed = agent2.parse_response(chunk, response(chunk, "A", ["on one line"]))
    assert parsed[0]["resolution"] == "A"
    assert parsed[0]["evidence_spans"] == ["on one line"]


def test_unresolved_question_has_no_evidence_spans(chunk) -> None:
    parsed = agent2.parse_response(chunk, response(chunk, "unresolved", []))
    assert parsed[0]["resolution"] == "unresolved"


@pytest.mark.parametrize(
    "resolution,spans",
    [
        ("A", []),
        ("unresolved", ["on one line"]),
        ("A", ["a span not in the requirement"]),
        ("A", ["on one line", "on one line"]),
        ("invalid", []),
    ],
)
def test_contrastive_resolution_rejects_invalid_or_unsupported_evidence(
    chunk, resolution, spans,
) -> None:
    with pytest.raises(agent2.Agent2Error):
        agent2.parse_response(chunk, response(chunk, resolution, spans))


@pytest.mark.parametrize("status,expected", [(400, "permanent"), (401, "permanent"),
                                           (404, "permanent"), (429, "transient"),
                                           (503, "transient")])
def test_qualification_transport_does_not_retry_permanent_statuses(status, expected) -> None:
    error = RuntimeError("synthetic provider error")
    error.status_code = status
    assert base.classify_exception(error)[0] == expected
