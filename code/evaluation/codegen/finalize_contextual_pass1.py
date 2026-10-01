#!/usr/bin/env python3
"""Assemble completed model evaluations into the final pass@1 table."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from contextual_pass1_common import (
    METHOD_ORDER,
    MODEL_ORDER,
    ExperimentError,
    atomic_json,
    model_slug,
    read_json,
    sha256_file,
    stable_hash,
    utc_now,
)


DISPLAY_METHOD = {
    "vanilla": "Vanilla LLM",
    "ask_or_assume": "Ask-or-Assume",
    "clarifygpt": "ClarifyGPT",
    "ours": "Ours",
}
DISPLAY_MODEL = {
    "gpt-5-mini": "GPT-5-mini",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gpt-5.5": "GPT-5.5",
    "grok-4.5": "Grok 4.5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            rate * (1 - rate) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - half, center + half]


def main() -> int:
    args = parse_args()
    completed: dict[str, dict] = {}
    sources: dict[str, dict[str, str]] = {}
    for model in MODEL_ORDER:
        path = args.evaluation_dir / model_slug(model) / "SUMMARY.json"
        if not path.exists():
            if args.allow_partial:
                continue
            raise ExperimentError(f"missing model evaluation: {model}")
        summary = read_json(path)
        if (
            not isinstance(summary, dict)
            or summary.get("schema_version")
            != "contextual-livecodebench-pass1-model-v1"
            or summary.get("model") != model
        ):
            raise ExperimentError(f"invalid model evaluation: {model}")
        completed[model] = summary
        sources[model] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "summary_identity": summary["summary_identity"],
        }

    rows = []
    for model in MODEL_ORDER:
        if model not in completed:
            continue
        for method in METHOD_ORDER:
            result = completed[model]["methods"][method]
            solved = int(result["solved"])
            total = int(result["tasks"])
            if total != 328:
                raise ExperimentError(f"{model}/{method}: task count is not 328")
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "solved": solved,
                    "tasks": total,
                    "pass_at_1": solved / total,
                    "wilson_95": wilson(solved, total),
                    "outcomes": result["outcomes"],
                }
            )

    final = {
        "schema_version": "contextual-livecodebench-pass1-final-v1",
        "completed_at": utc_now(),
        "complete": len(completed) == len(MODEL_ORDER),
        "model_order": list(MODEL_ORDER),
        "method_order": list(METHOD_ORDER),
        "models_completed": list(completed),
        "rows": rows,
        "sources": sources,
    }
    final["final_identity"] = stable_hash(final)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out_dir / "FINAL_SUMMARY.json", final)

    header = ["Model", *[DISPLAY_METHOD[m] for m in METHOD_ORDER]]
    lines = ["\t".join(header)]
    for model in MODEL_ORDER:
        if model not in completed:
            continue
        values = [DISPLAY_MODEL[model]]
        for method in METHOD_ORDER:
            result = completed[model]["methods"][method]
            values.append(
                f"{result['solved']}/328 "
                f"({100 * result['pass_at_1']:.2f}%)"
            )
        lines.append("\t".join(values))
    (args.out_dir / "table.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    latex = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Backbone & Vanilla LLM & Ask-or-Assume & ClarifyGPT & "
        + r"\model{} \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        if model not in completed:
            continue
        values = []
        for method in METHOD_ORDER:
            result = completed[model]["methods"][method]
            values.append(f"{100 * result['pass_at_1']:.2f}")
        latex.append(
            f"{DISPLAY_MODEL[model]} & " + " & ".join(values) + r" \\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (args.out_dir / "table.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
