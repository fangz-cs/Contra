# Optional coding-agent comparison harness

This harness compares coding-agent behavior with the clarification protocol.
It is separate from the CONTRA question selector and developer plugin.

The `native` arm uses the harness's normal action contract. `ask_permitted`
allows clarification before writing a program. `tag` retains the benchmark's
`[QUESTION]` and `[CODE]` output contract.

## Dependencies

Python 3.12 or newer is required for the optional OpenHands backend:

```bash
python -m pip install -e ".[harness]"
```

Supply a benchmark file locally with `--tasks`. No benchmark or run output is
included in this release.

## Free checks

```bash
python code/harness/run_harness.py \
  --tasks inputs/tasks.jsonl \
  --stub --arms native,ask_permitted --limit 4

python code/harness/run_harness.py \
  --tasks inputs/tasks.jsonl \
  --models qwen3.8-27b --arms native --dry-run
```

Stub mode does not call a model. Dry-run mode estimates cost without an API
call. Rates and uncalibrated token budgets must be checked against your
provider before a live run.

## Live execution

The explicit model settings are in [config.py](config.py). Credentials and
endpoint URLs come from the environment variables named there. Configure
native tool calling on your endpoint; a backend that rejects tool parameters
cannot run these arms.

Use `--preflight --confirm` for a real preflight call. A campaign also requires
`--confirm`. Review `--help`, the model budgets, and the judge calibration
requirements before starting.

The harness records request parameters, responses, usage, retries, timings,
and source revision. Completed cells are skipped on resume. Rows are flushed
and synced individually; a torn final line is recovered on load. Permanent
provider failures are not retried, and repeated failures trip a circuit breaker.

Generated outputs belong under the ignored results directories. They can
contain requirements, annotations, and responses, and must not be published
as part of this code-only repository.
