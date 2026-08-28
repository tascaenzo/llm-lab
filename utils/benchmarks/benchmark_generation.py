#!/usr/bin/env python3
"""Repeatable end-to-end benchmark for cached autoregressive generation."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


METRICS = re.compile(
    r"Inferenza cached: prefill (?P<prompt>\d+) token in (?P<prefill>[0-9.]+)s "
    r"\((?P<prefill_tps>[0-9.]+) token/s\), decode (?P<decode>\d+) token in "
    r"(?P<decode_seconds>[0-9.]+)s \((?P<decode_tps>[0-9.]+) token/s\)\."
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def run_once(arguments: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(arguments, capture_output=True, text=True, check=False)
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"generation failed with exit code {completed.returncode}:\n{completed.stderr}"
        )
    match = METRICS.search(completed.stderr)
    if match is None:
        raise RuntimeError("the CLI did not emit cached inference metrics")
    return {
        "wall_seconds": wall_seconds,
        "prompt_tokens": int(match.group("prompt")),
        "prefill_seconds": float(match.group("prefill")),
        "prefill_tokens_per_second": float(match.group("prefill_tps")),
        "decode_tokens": int(match.group("decode")),
        "decode_seconds": float(match.group("decode_seconds")),
        "decode_tokens_per_second": float(match.group("decode_tps")),
        "output": completed.stdout.rstrip("\n"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cli", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("--prompt", default="Ciao")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.tokens <= 0 or options.runs <= 0 or options.warmup < 0:
        parser.error("--tokens and --runs must be positive; --warmup cannot be negative")

    command = [
        str(options.cli),
        "model",
        "generate",
        str(options.checkpoint),
        str(options.tokenizer),
        str(options.tokens),
        options.prompt,
        "--backend",
        options.backend,
        "--seed",
        str(options.seed),
    ]
    for _ in range(options.warmup):
        run_once(command)

    runs = [run_once(command) for _ in range(options.runs)]
    outputs = {str(run["output"]) for run in runs}
    if len(outputs) != 1:
        raise RuntimeError("generation output changed across deterministic benchmark runs")
    decode_rates = [float(run["decode_tokens_per_second"]) for run in runs]
    prefill_rates = [float(run["prefill_tokens_per_second"]) for run in runs]
    wall_times = [float(run["wall_seconds"]) for run in runs]
    report = {
        "schema": "llm-lab-generation-benchmark-v1",
        "backend": options.backend,
        "command": command,
        "runs": options.runs,
        "warmup_runs": options.warmup,
        "prompt_tokens": runs[0]["prompt_tokens"],
        "decode_tokens": runs[0]["decode_tokens"],
        "decode_tokens_per_second": {
            "median": statistics.median(decode_rates),
            "minimum": min(decode_rates),
            "p95": percentile(decode_rates, 0.95),
        },
        "prefill_tokens_per_second": {
            "median": statistics.median(prefill_rates),
            "minimum": min(prefill_rates),
        },
        "wall_seconds": {
            "median": statistics.median(wall_times),
            "p95": percentile(wall_times, 0.95),
        },
        "deterministic_output": runs[0]["output"],
        "samples": runs,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
