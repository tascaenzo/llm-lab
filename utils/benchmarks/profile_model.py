#!/usr/bin/env python3
"""Per-operation backend profile of one Italiano-Base-75M training update.

The runtime benchmark measures a single operation at a single shape. A training
update is a fixed, known sequence of those operations, so the profile is the sum
of each measured shape times the number of times the update executes it.

The multiplicities below are read off src/model: lm_transformer_forward,
lm_transformer_backward, lm_output_head_*, lm_embedding_* and lm_trainer_step.

Usage:
    python3 utils/benchmarks/profile_model.py \
        --benchmark build/release/utils/benchmarks/runtime_benchmark
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    from .project_environment import configured_backend
except ImportError:  # Direct script execution.
    from project_environment import configured_backend


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_workloads(batch, sequence, hidden, heads, feed_forward, layers, vocabulary):
    """Returns (stage, operation, dimension flags, calls per update) tuples."""
    rows = batch * sequence
    head_dim = hidden // heads
    # Two micro-batches per update at the canonical gradient_accumulation_steps.
    micro = 2
    matrix = lambda r, k, c: ["--rows", str(r), "--inner", str(k), "--columns", str(c)]
    vector = lambda n: ["--elements", str(n)]
    attn = ["--batch", str(batch), "--sequence", str(sequence),
            "--query-heads", str(heads), "--kv-heads", str(heads),
            "--head-dim", str(head_dim)]
    norm = ["--rows", str(rows), "--columns", str(hidden)]
    ce = ["--rows", str(rows), "--columns", str(vocabulary)]

    w = []
    add = lambda stage, op, dims, calls: w.append((stage, op, dims, calls * micro))

    # ---- attention block, per layer ----
    add("attention", "rms_norm", norm, layers)
    add("attention", "matmul", matrix(rows, hidden, hidden), 4 * layers)
    add("attention", "rope", attn, 2 * layers)
    add("attention", "attention", attn, layers)
    add("attention", "add", vector(rows * hidden), layers)
    add("attention", "attention_backward", attn, layers)
    add("attention", "rope_backward", attn, 2 * layers)
    add("attention", "rms_norm_backward", norm, layers)
    # four linears, each: input gradient (transpose_right) + weight gradient
    add("attention", "matmul_transpose_right", matrix(rows, hidden, hidden), 4 * layers)
    add("attention", "matmul_transpose_left", matrix(rows, hidden, hidden), 4 * layers)
    add("attention", "accumulate", vector(hidden * hidden), 4 * layers)
    add("attention", "copy", vector(rows * hidden), layers)
    add("attention", "accumulate", vector(rows * hidden), 3 * layers)

    # ---- SwiGLU MLP, per layer ----
    add("mlp", "rms_norm", norm, layers)
    add("mlp", "matmul", matrix(rows, hidden, feed_forward), 2 * layers)
    add("mlp", "silu", vector(rows * feed_forward), layers)
    add("mlp", "multiply", vector(rows * feed_forward), layers)
    add("mlp", "matmul", matrix(rows, feed_forward, hidden), layers)
    add("mlp", "add", vector(rows * hidden), 2 * layers)
    add("mlp", "rms_norm_backward", norm, layers)
    add("mlp", "silu_backward", vector(rows * feed_forward), layers)
    add("mlp", "multiply", vector(rows * feed_forward), 2 * layers)
    add("mlp", "matmul_transpose_right", matrix(rows, hidden, feed_forward), layers)
    add("mlp", "matmul_transpose_left", matrix(rows, feed_forward, hidden), layers)
    add("mlp", "matmul_transpose_right", matrix(rows, feed_forward, hidden), 2 * layers)
    add("mlp", "matmul_transpose_left", matrix(rows, hidden, feed_forward), 2 * layers)
    add("mlp", "accumulate", vector(hidden * feed_forward), 3 * layers)
    add("mlp", "accumulate", vector(rows * hidden), 2 * layers)

    # ---- output head and loss ----
    add("lm_head", "matmul", matrix(rows, hidden, vocabulary), 1)
    add("lm_head", "cross_entropy_forward", ce, 1)
    add("lm_head", "cross_entropy_backward", ce, 1)
    add("lm_head", "matmul_transpose_right", matrix(rows, hidden, vocabulary), 1)
    add("lm_head", "matmul_transpose_left", matrix(rows, vocabulary, hidden), 1)
    add("lm_head", "accumulate", vector(hidden * vocabulary), 1)
    add("lm_head", "rms_norm", norm, 1)
    add("lm_head", "rms_norm_backward", norm, 1)

    # ---- embedding ----
    add("embedding", "gather", ["--rows", str(vocabulary), "--columns", str(hidden),
                                "--elements", str(rows)], 1)
    add("embedding", "scatter_add", ["--rows", str(vocabulary), "--columns", str(hidden),
                                     "--elements", str(rows)], 1)

    # ---- gradients and optimizer, once per update -----------------------
    # Keep the real parameter tensor boundaries. The runtime launches zero,
    # gradient-norm reductions and AdamW once per parameter rather than over
    # one imaginary flat 75M-element vector.
    parameter_shapes = [
        (vocabulary, hidden, 1),          # token embedding
        (hidden, vocabulary, 1),          # output head
        (1, hidden, 2 * layers + 1),      # two norms/layer plus final norm
        (hidden, hidden, 4 * layers),     # Q, K, V and attention output
        (hidden, feed_forward, 2 * layers),
        (feed_forward, hidden, layers),
    ]
    parameters = sum(rows * columns * count for rows, columns, count in parameter_shapes)
    parameter_tensor_count = sum(count for _, _, count in parameter_shapes)
    for rows, columns, count in parameter_shapes:
        elements = rows * columns
        w.append(("optimizer", "zero", vector(elements), count))
        w.append(("optimizer", "adamw", vector(elements), count))
        w.append(("gradient_norm", "reduce_mean_square",
                  ["--rows", str(rows), "--columns", str(columns)], count))
        w.append(("gradient_norm", "reduce_sum",
                  ["--rows", "1", "--columns", str(rows)], count))
    # One scalar scale and accumulation follow each parameter reduction; the
    # destination scalar is cleared once at the start of the norm calculation.
    w.append(("gradient_norm", "scale", vector(1), parameter_tensor_count))
    w.append(("gradient_norm", "accumulate", vector(1), parameter_tensor_count))
    w.append(("gradient_norm", "fill", vector(1), 1))
    return w, parameters


def measure(benchmark, backend, operation, dims, iterations, sample_ms):
    command = [benchmark, "--backend", backend, "--batched", "--operations", operation,
               "--warmup", "2", "--iterations", str(iterations),
               "--sample-ms", str(sample_ms), "--format", "jsonl"] + dims
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, end="")
        return None
    output = completed.stdout
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") == "result" and record.get("operation") == operation:
            gpu = record.get("gpu_median_seconds") or 0.0
            return gpu if gpu > 0.0 else record["median_seconds"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--feed-forward", type=int, default=1608)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--vocabulary", type=int, default=32008)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--sample-ms", type=int, default=20)
    arguments = parser.parse_args()
    if arguments.backend is None:
        try:
            arguments.backend = configured_backend(
                PROJECT_ROOT, default="cpu", allowed=("cpu", "metal", "cuda")
            )
        except ValueError as error:
            parser.error(str(error))

    workloads, parameters = build_workloads(
        arguments.batch, arguments.sequence, arguments.hidden, arguments.heads,
        arguments.feed_forward, arguments.layers, arguments.vocabulary)
    print(f"Backend: {arguments.backend} | modello: {parameters/1e6:.2f}M parametri, "
          f"{len(workloads)} forme distinte da misurare\n", file=sys.stderr)

    cache = {}
    by_stage = defaultdict(float)
    by_operation = defaultdict(float)
    rows = []
    for index, (stage, operation, dims, calls) in enumerate(workloads, 1):
        key = (operation, tuple(dims))
        if key not in cache:
            print(f"  [{index}/{len(workloads)}] {operation} {' '.join(dims)}",
                  file=sys.stderr)
            cache[key] = measure(arguments.benchmark, arguments.backend, operation, dims,
                                 arguments.iterations, arguments.sample_ms)
        seconds = cache[key]
        if seconds is None:
            print(f"      non misurabile, saltata", file=sys.stderr)
            continue
        total = seconds * calls
        by_stage[stage] += total
        by_operation[operation] += total
        rows.append((stage, operation, " ".join(d for d in dims if not d.startswith("--")),
                     calls, seconds * 1e3, total * 1e3))

    grand_total = sum(by_stage.values())
    if grand_total <= 0.0:
        raise SystemExit("Nessuna operazione misurata: verifica che il backend GPU sia disponibile.")
    print(f"\n{'='*78}\nSTIMA GPU DI UN UPDATE — somma {grand_total*1e3:.1f} ms\n{'='*78}")
    print(f"\n{'Stadio':<14}{'ms':>10}{'%':>8}")
    for stage, seconds in sorted(by_stage.items(), key=lambda kv: -kv[1]):
        print(f"{stage:<14}{seconds*1e3:>10.1f}{seconds/grand_total*100:>7.1f}%")
    print(f"\n{'Operazione':<26}{'ms':>10}{'%':>8}")
    for operation, seconds in sorted(by_operation.items(), key=lambda kv: -kv[1]):
        if seconds / grand_total < 0.002:
            continue
        print(f"{operation:<26}{seconds*1e3:>10.1f}{seconds/grand_total*100:>7.1f}%")
    print(f"\n{'Stadio':<11}{'Operazione':<25}{'forma':<22}{'n':>5}{'ms/call':>10}{'tot ms':>10}")
    for row in sorted(rows, key=lambda r: -r[5])[:22]:
        print(f"{row[0]:<11}{row[1]:<25}{row[2]:<22}{row[3]:>5}{row[4]:>10.3f}{row[5]:>10.1f}")


if __name__ == "__main__":
    main()
