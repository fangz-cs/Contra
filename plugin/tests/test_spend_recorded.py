"""The envelope must yield its reply and its price, and never eat a reply."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hooks"))
import verify_disclosures as vd  # noqa: E402


def envelope(result: str, cost: float = 0.42, turns: int = 7) -> str:
    return json.dumps({
        "type": "result", "subtype": "success", "result": result,
        "total_cost_usd": cost, "num_turns": turns,
        "usage": {"cache_creation_input_tokens": 11, "cache_read_input_tokens": 22,
                  "input_tokens": 2, "output_tokens": 33},
    })


def setup_function(_):
    vd.SPEND.clear()


def test_reply_comes_back_and_the_price_is_recorded():
    out = vd._unwrap(envelope("```json\n{\"decisions\": []}\n```"), "probe", None)
    assert "decisions" in out
    assert len(vd.SPEND) == 1
    row = vd.SPEND[0]
    assert row["stage"] == "probe"
    assert row["cost_usd"] == 0.42
    assert row["api_calls"] == 7
    assert row["last_msg_output"] == 33


def test_plain_text_passes_through():
    """An older CLI, or an error path, prints the bare reply."""
    raw = "```json\n{\"items\": []}\n```"
    assert vd._unwrap(raw, "extract", "haiku") == raw
    assert vd.SPEND == []


def test_a_model_answer_named_result_is_not_unwrapped():
    """The reply is JSON too; only the CLI's own marker may identify an envelope."""
    raw = json.dumps({"result": "not the envelope", "decisions": [{"gap": "x"}]})
    assert vd._unwrap(raw, "probe", None) == raw
    assert vd.SPEND == []


def test_an_empty_result_is_empty_not_none():
    assert vd._unwrap(envelope(""), "build", None) == ""


def test_the_reply_survives_a_missing_usage_block():
    env = json.loads(envelope("hi"))
    del env["usage"]
    assert vd._unwrap(json.dumps(env), "build", None) == "hi"
    assert vd.SPEND[0]["last_msg_output"] == 0
