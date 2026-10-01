# CONTRA developer plugin

After a code change, CONTRA reports decisions that the request did not state.
It distinguishes work beyond the request, work left undone, and open choices
settled by the implementation. A confirmed open choice has a discriminating
test that passes on the implemented reading and fails on a runnable rival.

## Installation

Run from the repository root:

```bash
python -m pip install -e ".[plugin]"
claude --plugin-dir ./plugin
```

The root marketplace descriptor also supports local installation:

```bash
claude plugin marketplace add /path/to/contra-checkout
claude plugin install contra@contra
```

Requirements:

- Python 3.10 or newer and the `claude` CLI with an authenticated account.
- `git` for deduplicated prompt-time snapshots.
- `pytest` for executable verification.
- Linux and `bubblewrap` for isolated test execution.

The `pytest` dependency runs plugin-generated discriminating tests. No
development test suite is included in this release.

Without `git`, the baseline uses capped inline source. Without a working
sandbox, verification is reported as unavailable by default. Source reporting
covers the configured suffixes; executable verification currently supports
Python.

## Workflow

1. `UserPromptSubmit` records the current source tree.
2. The synchronous `Stop` hook compares the prompt-time baseline with the
   changed tree. It extracts admitted choices and reviews the request and diff
   independently. That review does not see the completion report.
3. The hook merges and ranks decisions, then returns the report.
4. Background verification builds a test and rival patch for open choices.
   Confirmed tests can wake the assistant for saving.

New files, staged edits, and changes made after the prompt are included.
Earlier uncommitted changes are not treated as this turn's work. A missing
baseline does not fall back to the full working tree.

The asynchronous path is part of the supported plugin hook protocol.

## Configuration

Settings resolve in this order:

1. `CONTRA_<SETTING>` environment variable.
2. `CONTRA_CONFIG`, or `$XDG_CONFIG_HOME/contra/config.json`.
3. The built-in default.

Without `XDG_CONFIG_HOME`, the file defaults to `~/.config/contra/config.json`.
Invalid values produce a warning before using the default.

```bash
python plugin/hooks/config.py
```

| Setting | Default | Purpose |
|---|---|---|
| `lang` | locale | English or Chinese report frame |
| `max_items` | `8` | Report size, balanced across decision kinds |
| `extract_model` | `haiku` | Extract choices admitted in the completion report |
| `probe_model` | CLI default | Independent request-and-diff review |
| `build_model` | CLI default | Discriminating test and rival patch |
| `timeout_extract` | `180` | Model extraction timeout in seconds |
| `timeout_build` | `300` | Test-building timeout in seconds |
| `test_timeout` | `300` | Test timeout in seconds |
| `source_suffixes` | configured source languages | Files included in the baseline |
| `runnable_suffixes` | `.py` | Files eligible for executable evidence |
| `skip_unchanged_ast` | `true` | Skip Python edits with unchanged syntax trees |
| `max_files` | `2000` | Snapshot file limit |
| `max_bytes` | `524288` | Snapshot per-file byte limit |
| `sandbox` | `required` | Refuse host execution when isolation is unavailable |
| `merge_threshold` | `0.28` | Conservative character-bigram overlap threshold |

`CONTRA_STATE_DIR` relocates snapshots, hand-offs, and the ledger. Its default
is under `$XDG_STATE_HOME/contra`. This state can contain requests and source
text, so keep it private and outside the publication repository.

`CONTRA_INSIDE` marks plugin-started subprocesses to prevent recursive hooks.
Do not set it for normal use.

## Evidence and execution

Only a passing implemented-reading test and a failing runnable rival confirm
a choice. A rival that cannot be collected is unverifiable. A test that passes
under both readings is `not_separated`, not proof of equivalence.

Model-written tests run in a copied workspace under `bubblewrap`, without
network access or a view of the real working tree. An interpreter installed
under the home directory may need its own installation path mounted.
`sandbox=allow_host` explicitly permits host execution when isolation is
unavailable; it is not the default.

Model calls use the authenticated CLI and may incur charges. They receive the
request and relevant source changes. Use the plugin only where sending those
inputs to the configured provider is authorized.

## Layout

| Path | Purpose |
|---|---|
| [hooks/record_baseline.py](hooks/record_baseline.py) | Prompt-time baseline hook |
| [hooks/baseline.py](hooks/baseline.py) | Snapshot and changed-tree comparison |
| [hooks/verify_disclosures.py](hooks/verify_disclosures.py) | Reporting and verification hooks |
| [scripts/discriminate.py](scripts/discriminate.py) | Run a test on both readings |
| [scripts/exec_patch.py](scripts/exec_patch.py) | Apply rival edits in a copy |
| [scripts/rank.py](scripts/rank.py) | Rank and render decisions |
| [scripts/ledger.py](scripts/ledger.py) | Cumulative view across turns |
| [skills/clarify-before-coding/SKILL.md](skills/clarify-before-coding/SKILL.md) | Decision-reporting skill |

Only the active plugin implementation is included.

## License

[Apache License 2.0](LICENSE).
