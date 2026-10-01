# CONTRA

**Discovering and Qualifying Behavior-Changing Questions for Selective
Clarification in LLM Code Generation**

This repository contains the available original CONTRA method implementations
and the active developer plugin.

## Release scope

The release includes decision relevance, contrastive resolution, frozen
question-plan construction, adaptive question selection, and the plugin.

It contains no development test suite, CI workflows, publication-check scripts,
standalone evaluation or comparison tools, or supplemental reference components.
Benchmark tasks, annotations, candidate pools, frozen plans, experiment outputs,
transcripts, model weights, paper archives, and machine-specific provenance are
also excluded.

The supplied source snapshot did not include the original full offline
discovery or multi-program behavioral-verification drivers. They are not part
of this release, and no replacement implementations are included. Internal
`RITE` and `ForkCast` names remain where they preserve original wire formats.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For the plugin, install its executable-verification dependency:

```bash
python -m pip install -e ".[plugin]"
```

## Layout

| Path | Purpose |
|---|---|
| [code/pipeline/](code/pipeline/) | Original relevance, contrastive resolution, and frozen question-plan implementations |
| [code/selector/](code/selector/) | Original dialogue-conditioned adaptive selector and a public Chat transport |
| [plugin/](plugin/) | Developer plugin, hooks, skill, and discriminating-test implementation |
| [configs/](configs/) | Credential-free configuration template |

## Retained method implementations

### Qualification and frozen plans

The original scripts separate two judgments:

- [run_agent1_decision_relevance.py](code/pipeline/scripts/run_agent1_decision_relevance.py)
  checks whether the question concerns required program behavior.
- [run_agent2_contrastive_resolution.py](code/pipeline/scripts/run_agent2_contrastive_resolution.py)
  checks whether the requirement resolves the choice. A resolved judgment must
  cite verbatim evidence; an unresolved judgment must provide no evidence span.

[build_rite_v3_question_plan.py](code/pipeline/scripts/build_rite_v3_question_plan.py)
consumes existing qualification and behavioral-verification outputs. It retains
relevant, unresolved candidates with at least one stable behavioral difference,
preserves their discovery order, and binds the frozen queue to its inputs.
Behavioral-verification outputs must be supplied separately.

### Adaptive question selection

[run_interactive_question_selector.py](code/selector/run_interactive_question_selector.py)
selects one immutable candidate ID or stops. It replays selected question text
verbatim. The selector sees the requirement, dialogue, and remaining queue,
not benchmark annotations. Clarification is capped at five question rounds.

Shared question-plan helpers retain hash-binding and gold-blindness checks.
The public transport uses OpenAI-compatible Chat Completions. A custom transport
can be supplied with `--transport`; use the tool schema its endpoint supports.

## Running the original stages

All tasks and intermediate artifacts must be supplied locally. The repository
does not download or bundle any dataset.

Configure credentials in your shell, not in source files. The selector accepts
`CONTRA_API_KEY`, `CLARIFY_API_KEY`, or `OPENAI_API_KEY`, in that order. The
qualification configuration names its credential variables explicitly.
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

The original selector uses the ClarifyCodeBench interaction runtime. Supply an
authorized external checkout containing the `clarifycodebench` package, including
its `prompts`, `interact`, `extract`, and `llm` modules. That dependency is not
vendored here. Provide its parent directory with `--release`.

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

## Developer plugin

See [plugin/README.md](plugin/README.md) for installation and configuration.
The normal plugin path records a prompt-time baseline, surfaces decisions after
a code change, and verifies open choices in the background. This post-change
workflow is a practical adaptation of the clarification method.

The marketplace descriptor points to the plugin subdirectory. The plugin
manifest contains no personal author or contact fields.

## License

The included code is distributed under the [Apache License 2.0](LICENSE).
External dependencies retain their own licenses and distribution terms. This
core-only release is not a complete experimental reproduction package.
