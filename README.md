# CONTRA

**Discovering and Qualifying Behavior-Changing Questions for Selective
Clarification in LLM Code Generation**

CONTRA discovers possible clarification questions, qualifies questions whose
rival answers change required program behavior, and selects questions using the
dialogue so far. This repository also includes a developer plugin that exposes
unstated decisions after a code change and checks open choices with
discriminating tests.

## Release scope

This is a code-only release. It contains no benchmark tasks, annotations,
candidate pools, frozen question plans, experiment outputs, transcripts,
model weights, paper archives, or machine-specific provenance.

The original qualification and adaptive-selector implementations are retained.
The supplied source snapshot did not include the full offline discovery or
multi-program behavioral-verification campaign drivers. The `contra` package
provides small reference components for their discovery control flow and
behavioral voting rule. These components use caller-supplied generators,
semantic reviewers, validators, and isolated executors. They do not claim to
recreate the original experiments.

Historical server launchers, supervisors, private gateway adapters, and
run-specific finalizers are not included. Internal `RITE` and `ForkCast` schema
names remain where changing them would break input and result compatibility.

## Installation

Python 3.10 or newer is required. The plugin's executable verification requires
Linux, `pytest`, and `bubblewrap`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The optional OpenHands comparison harness requires Python 3.12 or newer:

```bash
python -m pip install -e ".[harness]"
```

## Layout

| Path | Purpose |
|---|---|
| [contra/](contra/) | Provider-independent discovery and behavioral voting components |
| [code/pipeline/](code/pipeline/) | Original relevance, contrastive resolution, question-plan, and scoring implementations |
| [code/selector/](code/selector/) | Original dialogue-conditioned adaptive selector and a public Chat transport |
| [code/evaluation/](code/evaluation/) | Clarification scoring and code-generation evaluation utilities |
| [code/harness/](code/harness/) | Optional coding-agent comparison harness |
| [plugin/](plugin/) | Developer plugin, hooks, skill, and discriminating-test implementation |
| [configs/](configs/) | Credential-free configuration template |
| [tests/](tests/) | Synthetic, offline method and release checks |

## Method

### 1. Question discovery

`contra.discover_questions` combines context-aware and independent generation
in each round. A caller-supplied reviewer merges overlapping questions.
Discovery stops after two consecutive rounds add no retained candidate, or
when `max_rounds` is reached. Two requirement completions are regenerated for
each retained question after discovery.

Candidate IDs must remain stable during review. The reviewer must keep an
existing ID for a merged question rather than create a new one. The completion
callback receives only the requirement and question. It must produce compatible
completions that differ only in the answer to that question. Semantic validity
is the responsibility of the supplied callbacks.

### 2. Question qualification

The original scripts separate two judgments:

- [run_agent1_decision_relevance.py](code/pipeline/scripts/run_agent1_decision_relevance.py)
  checks whether the question concerns required program behavior.
- [run_agent2_contrastive_resolution.py](code/pipeline/scripts/run_agent2_contrastive_resolution.py)
  checks whether the requirement resolves the choice. A resolved judgment must
  cite verbatim evidence; an unresolved judgment must provide no evidence span.

`contra.verify_behavior` implements the execution criterion. Supply three
independently generated programs for each completion and at least seven
distinct inputs valid under both completions. On each input, an output is
stable if at least two successful executions in that group agree. A question
passes when the groups have different stable outputs on at least one input.
All execution results are retained in the returned evidence object.

The reference verifier does not execute code itself. Supply an isolated
executor and a common-input validator. Crashes and timeouts do not vote. A
failed verification is absence of evidence, not evidence of equivalence.

Only relevant, unresolved questions with a behavioral witness belong in the
qualified queue.

### 3. Adaptive question selection

[run_interactive_question_selector.py](code/selector/run_interactive_question_selector.py)
selects one immutable candidate ID or stops. It replays selected question text
verbatim. The selector sees the requirement, dialogue, and remaining queue,
not benchmark annotations. Clarification is capped at five question rounds.

The frozen-plan builders retain the original hash-binding and gold-blindness
checks. The public transport uses OpenAI-compatible Chat Completions. A custom
transport can be supplied with `--transport`; use the tool schema that its
endpoint supports.

## Running the original stages

All tasks and intermediate artifacts must be supplied locally. The repository
does not download or bundle any dataset.

Configure credentials in your shell, not in source files. The selector accepts
`CONTRA_API_KEY`, `CLARIFY_API_KEY`, or `OPENAI_API_KEY`, in that order. The
qualification configuration names its credential variables explicitly.
Clarification audit utilities use `CONTRA_BASE_URL` and `CONTRA_API_KEY`.
The [.env.example](.env.example) file contains names and empty placeholders only.

Edit a local copy of [agent_models.yaml](configs/agent_models.yaml) for your
model and supported request parameters. Preparing Agent 1 performs no API calls:

```bash
python code/pipeline/scripts/run_agent1_decision_relevance.py \
  --input-jsonl inputs/candidates.jsonl \
  --models-config configs/agent_models.yaml \
  --model api-model \
  --output-dir outputs/agent1 \
  --prepare-only
```

Live qualification requires `--confirm-live`. Use each script's `--help` for
its input contract. Agent 2 consumes Agent 1's results, manifest, and completion
marker. Its expected selected count must be supplied for the input you use.

Interactive benchmark evaluation also needs an authorized external
ClarifyCodeBench checkout containing the `clarifycodebench` package, including
its `prompts`, `interact`, `extract`, `llm`, and `metrics` modules. That dependency
is not vendored here. Provide its parent directory with `--release`.

Validate a supplied dataset and frozen question plan without an API call:

```bash
python code/selector/run_interactive_question_selector.py \
  --data inputs/tasks.jsonl \
  --plan inputs/question-plan.json \
  --release /path/to/clarifycodebench_release \
  --model your-model-id \
  --out outputs/selector/results.jsonl \
  --dry-run
```

Replace `--dry-run` with `--confirm` to execute. Use
`--model-temperature none` if the provider rejects a temperature parameter.
The selector resumes compatible completed tasks by default and rejects
changed manifests.

Code-generation evaluation additionally requires an external LiveCodeBench
checkout and locally supplied test sources. The retained utilities document
their required paths in `--help`. The code-generation utility requires `--confirm` and accepts an
explicit `--api-key-file`; keep that file outside the publication repository.
Do not start live runs before reviewing the request parameters and cost.

## Developer plugin

See [plugin/README.md](plugin/README.md) for installation and configuration.
The normal plugin path records a prompt-time baseline, surfaces decisions after
a code change, and verifies open choices in the background. This post-change
workflow is a practical adaptation, not the offline benchmark protocol above.

The marketplace descriptor points to the plugin subdirectory. The plugin
manifest contains no personal author or contact fields.

## Verification

```bash
python -m pytest -q
python scripts/check_release.py
python -m build
```

Tests use synthetic in-memory inputs, temporary files, and mocked model calls.
They do not require benchmark data or paid API access. Checks cover the
discovery stopping rule, the three-program and seven-input thresholds,
contrastive evidence validation, frozen-plan contracts, selector behavior,
plugin hooks, and publication hygiene.

The release checker reports paths and categories, not matched secret text.
Its pattern checks complement manual review; they are not proof that arbitrary
new content contains no personal information. Datasets and generated artifacts
are also excluded by [.gitignore](.gitignore).

## License

The included code is distributed under the [Apache License 2.0](LICENSE).
External benchmark packages and test sources retain their own licenses and
distribution terms. This code-only release does not reproduce the paper's
numerical results without the original inputs and run configurations.
