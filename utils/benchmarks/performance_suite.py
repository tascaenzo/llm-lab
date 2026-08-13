#!/usr/bin/env python3
"""Run representative CPU/Metal benchmarks and compare them with a local baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from compare_results import compare, load_results, result_key


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "build" / "release" / "utils" / "benchmarks" / "runtime_benchmark"
)

ALL_OPERATIONS = {
    "zero",
    "fill",
    "copy",
    "add",
    "multiply",
    "scale",
    "accumulate",
    "reduce_sum",
    "reduce_max",
    "reduce_mean_square",
    "matmul",
    "matmul_transpose_left",
    "matmul_transpose_right",
    "gather",
    "scatter_add",
    "silu",
    "silu_backward",
    "rms_norm",
    "rms_norm_backward",
    "rope",
    "rope_backward",
    "attention",
    "attention_backward",
    "softmax",
    "cross_entropy_forward",
    "cross_entropy_backward",
    "adamw",
}

METAL_OPERATIONS = ALL_OPERATIONS


SCENARIOS: Sequence[Tuple[str, Sequence[str]]] = (
    (
        "small-latency-f32",
        (
            "--operations",
            "all",
            "--elements",
            "65536",
            "--rows",
            "64",
            "--columns",
            "128",
            "--inner",
            "64",
        ),
    ),
    (
        "vector-throughput-f32",
        (
            "--operations",
            "zero,fill,copy,add,multiply,scale,accumulate,"
            "silu,silu_backward,adamw",
            "--elements",
            "8388608",
        ),
    ),
    (
        "row-throughput-f32",
        (
            "--operations",
            "reduce_sum,reduce_max,reduce_mean_square,gather,scatter_add,rms_norm,"
            "rms_norm_backward,softmax,cross_entropy_forward,cross_entropy_backward",
            "--rows",
            "512",
            "--columns",
            "2048",
            "--inner",
            "256",
        ),
    ),
    (
        "matmul-square-f32",
        (
            "--operations",
            "matmul,matmul_transpose_left,matmul_transpose_right",
            "--rows",
            "512",
            "--columns",
            "512",
            "--inner",
            "512",
        ),
    ),
    (
        "transformer-kernels-f32",
        (
            "--operations",
            "rope,rope_backward,attention,attention_backward",
            "--batch",
            "1",
            "--sequence",
            "128",
            "--query-heads",
            "8",
            "--kv-heads",
            "2",
            "--head-dim",
            "64",
        ),
    ),
    (
        "matmul-transformer-f32",
        (
            "--operations",
            "matmul",
            "--rows",
            "512",
            "--columns",
            "4096",
            "--inner",
            "1024",
        ),
    ),
)


SMOKE_SCENARIOS: Sequence[Tuple[str, Sequence[str]]] = (
    (
        "smoke",
        (
            "--operations",
            "add,matmul",
            "--elements",
            "4096",
            "--rows",
            "16",
            "--columns",
            "16",
            "--inner",
            "16",
        ),
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a representative benchmark suite for every runtime kernel."
        )
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, help="save the current JSONL result")
    parser.add_argument("--baseline", type=Path, help="compare against this JSONL baseline")
    parser.add_argument(
        "--max-regression-percent",
        type=float,
        default=10.0,
        help="fail when median time regresses by more than this percentage (default: 10)",
    )
    parser.add_argument(
        "--noise-percent",
        type=float,
        default=3.0,
        help="changes inside this range are reported as stable (default: 3)",
    )
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def common_arguments(smoke: bool) -> List[str]:
    if smoke:
        return [
            "--format",
            "jsonl",
            "--backend",
            "all",
            "--threads",
            "auto",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--sample-ms",
            "1",
        ]
    return [
        "--format",
        "jsonl",
        "--backend",
        "all",
        "--threads",
        "auto",
        "--warmup",
        "5",
        "--iterations",
        "40",
        "--sample-ms",
        "30",
    ]


def selected_operations(arguments: Sequence[str]) -> set[str]:
    try:
        operations = arguments[arguments.index("--operations") + 1]
    except (IndexError, ValueError) as error:
        raise ValueError("scenario has no --operations argument") from error
    return ALL_OPERATIONS if operations == "all" else set(operations.split(","))


def expected_result_count(
    scenarios: Sequence[Tuple[str, Sequence[str]]], metal_available: bool
) -> int:
    total = 0
    for _, arguments in scenarios:
        operations = selected_operations(arguments)
        total += len(operations)
        if metal_available:
            total += len(operations & METAL_OPERATIONS)
    return total


def parse_json_line(line: str, scenario: str, line_number: int) -> Dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{scenario}:{line_number}: invalid benchmark JSON: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{scenario}:{line_number}: benchmark record must be an object")
    record["scenario"] = scenario
    return record


def format_duration(seconds: float) -> str:
    if seconds >= 0.001:
        return f"{seconds * 1000.0:.3f} ms"
    if seconds >= 0.000001:
        return f"{seconds * 1000000.0:.3f} us"
    return f"{seconds * 1000000000.0:.1f} ns"


def print_progress(record: Dict[str, Any], completed: int, total: int) -> None:
    width = 24
    filled = min(width, int(width * completed / total)) if total > 0 else 0
    bar = "#" * filled + "-" * (width - filled)
    percent = completed * 100.0 / total if total > 0 else 0.0
    throughput = f'{float(record["throughput"]):.3f} {record["throughput_unit"]}'
    print(
        f"[{bar}] {completed:>2}/{total:<2} {percent:>5.1f}%  "
        f'{str(record.get("backend", "cpu")).upper():<5} '
        f'{str(record["operation"]):<24} {str(record.get("dtype", "f32")):<4} '
        f'| {format_duration(float(record["median_seconds"])):>10} | {throughput}',
        flush=True,
    )


def run_scenarios(
    benchmark: Path, scenarios: Sequence[Tuple[str, Sequence[str]]], smoke: bool
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    common = common_arguments(smoke)
    completed_results = 0
    expected_results: Optional[int] = None
    for index, (name, arguments) in enumerate(scenarios, start=1):
        scenario_started = time.monotonic()
        scenario_results = 0
        print(f"\nScenario {index}/{len(scenarios)}: {name}", flush=True)
        process = subprocess.Popen(
            [str(benchmark), *common, *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdout is None or process.stderr is None:
            process.terminate()
            raise RuntimeError(f"{name}: unable to capture benchmark output")
        try:
            for line_number, line in enumerate(process.stdout, start=1):
                if not line.strip():
                    continue
                record = parse_json_line(line, name, line_number)
                records.append(record)
                if record.get("type") == "metadata" and expected_results is None:
                    expected_results = expected_result_count(
                        scenarios, bool(record.get("metal_available", False))
                    )
                    metal = "disponibile" if record.get("metal_available") else "non disponibile"
                    print(
                        f'Sistema: {record.get("os", "?")} / '
                        f'{record.get("architecture", "?")} | '
                        f'CPU: {record.get("detected_hardware_threads", "?")} thread | '
                        f"Metal: {metal} | Test previsti: {expected_results}",
                        flush=True,
                    )
                elif record.get("type") == "result":
                    completed_results += 1
                    scenario_results += 1
                    print_progress(record, completed_results, expected_results or completed_results)
        except (KeyError, TypeError, ValueError):
            process.terminate()
            process.wait()
            raise
        error_output = process.stderr.read().strip()
        return_code = process.wait()
        if return_code != 0:
            detail = error_output or "unknown error"
            raise RuntimeError(f"{name}: benchmark failed: {detail}")
        if scenario_results == 0:
            raise ValueError(f"{name}: benchmark produced no result records")
        print(
            f"Scenario completato: {scenario_results} test in "
            f"{time.monotonic() - scenario_started:.1f} secondi.",
            flush=True,
        )
    return records


def result_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if record.get("type") == "result"]


def dimensions_text(dimensions: Dict[str, Any]) -> str:
    if "elements" in dimensions:
        return str(dimensions["elements"])
    if "inner" in dimensions:
        return f'{dimensions["rows"]}x{dimensions["inner"]}x{dimensions["columns"]}'
    if "batch" in dimensions:
        return (
            f'B{dimensions["batch"]}xS{dimensions["sequence"]}x'
            f'H{dimensions["query_heads"]}xD{dimensions["head_dimension"]}'
        )
    return f'{dimensions["rows"]}x{dimensions["columns"]}'


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def current_results_map(records: Iterable[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    results: Dict[Any, Dict[str, Any]] = {}
    for record in result_records(records):
        key = result_key(record)
        if key in results:
            raise ValueError(f"duplicate current benchmark result: {key}")
        results[key] = record
    return results


def comparison_status(
    change_percent: float, noise_percent: float, regression_percent: float
) -> str:
    if change_percent < -noise_percent:
        return "MIGLIORATO"
    if change_percent > regression_percent:
        return "REGRESSIONE"
    if change_percent > noise_percent:
        return "PIU' LENTO"
    return "STABILE"


def print_comparison(report: Dict[str, Any], noise_percent: float) -> None:
    comparisons = report.get("comparisons", [])
    print("\nConfronto con la baseline")
    print(
        f'{"Scenario":<26} {"HW":<6} {"Kernel":<24} {"Tipo":<5} '
        f'{"Baseline":>12} {"Attuale":>12} {"Variazione":>12} {"Esito":>12}'
    )
    print("-" * 121)
    for item in comparisons:
        change = float(item["change_percent"])
        status = comparison_status(
            change, noise_percent, float(report["maximum_regression_percent"])
        )
        print(
            f'{item["scenario"]:<26} {item["backend"]:<6} {item["operation"]:<24} '
            f'{item["dtype"]:<5} {format_duration(float(item["baseline_seconds"])):>12} '
            f'{format_duration(float(item["current_seconds"])):>12} '
            f'{change:>+10.2f}% {status:>12}'
        )
    print(
        f'\nConfrontati {report["compared_records"]} risultati: '
        f'{len(report["regressions"])} regressioni oltre soglia, '
        f'{len(report["missing"])} risultati mancanti.'
    )


def print_matmul_summary(results: Sequence[Dict[str, Any]]) -> None:
    groups: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = {}
    order: List[Tuple[str, str, str]] = []
    for record in results:
        scenario = str(record.get("scenario", ""))
        if record.get("operation") != "matmul" or not scenario.startswith("matmul-"):
            continue
        key = (
            scenario,
            str(record.get("dtype", "f32")),
            dimensions_text(record["dimensions"]),
        )
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][str(record.get("backend", "cpu"))] = record

    if not order:
        return
    print("\nMatmul principali")
    print(f'{"Scenario":<25} {"Tipo":<5} {"Forma":<18} {"CPU":>14} {"Metal":>14} {"Vincitore":>15}')
    print("-" * 96)
    for scenario, dtype, shape in order:
        backends = groups[(scenario, dtype, shape)]
        cpu = backends.get("cpu")
        metal = backends.get("metal")
        cpu_text = f'{float(cpu["throughput"]):.1f} GFLOP/s' if cpu else "-"
        metal_text = f'{float(metal["throughput"]):.1f} GFLOP/s' if metal else "-"
        if cpu is not None and metal is not None:
            ratio = float(metal["throughput"]) / float(cpu["throughput"])
            winner = f"Metal {ratio:.2f}x" if ratio >= 1.0 else f"CPU {1.0 / ratio:.2f}x"
        elif metal is not None:
            winner = "Metal"
        else:
            winner = "CPU"
        label = scenario[len("matmul-") :]
        print(
            f"{label:<25} {dtype:<5} {shape:<18} {cpu_text:>14} "
            f"{metal_text:>14} {winner:>15}"
        )


def print_final_summary(
    records: Sequence[Dict[str, Any]],
    elapsed_seconds: float,
    comparison_report: Optional[Dict[str, Any]],
    noise_percent: float,
    output_path: Optional[Path],
) -> None:
    results = result_records(records)
    metadata = next(
        (record for record in records if record.get("type") == "metadata"), {}
    )
    cpu_count = sum(record.get("backend") == "cpu" for record in results)
    metal_count = sum(record.get("backend") == "metal" for record in results)
    metal_devices = sorted(
        {
            str(record.get("device"))
            for record in results
            if record.get("backend") == "metal" and record.get("device")
        }
    )
    unstable_count = sum(
        float(record.get("coefficient_of_variation", 0.0)) > 0.10 for record in results
    )
    has_failure = bool(
        comparison_report
        and (comparison_report.get("regressions") or comparison_report.get("missing"))
    )

    print("\n" + "=" * 72)
    print("RIEPILOGO FINALE")
    print("=" * 72)
    print(f'Esito: {"REGRESSIONE RILEVATA" if has_failure else "COMPLETATO"}')
    print(
        f'Test completati: {len(results)} | CPU: {cpu_count} | Metal: {metal_count} '
        f"| Durata: {elapsed_seconds:.1f} secondi"
    )
    print(
        f'Sistema: {metadata.get("os", "?")} / {metadata.get("architecture", "?")} '
        f'| CPU: {metadata.get("detected_hardware_threads", "?")} thread'
    )
    if metal_devices:
        print(f'GPU: {", ".join(metal_devices)}')
    print(
        f"Stabilita': {len(results) - unstable_count}/{len(results)} risultati con "
        "variabilita' entro il 10%"
    )

    if comparison_report is None:
        print("Confronto: baseline non fornita; questa esecuzione puo' essere salvata come baseline.")
    else:
        counts = {"MIGLIORATO": 0, "STABILE": 0, "PIU' LENTO": 0, "REGRESSIONE": 0}
        for item in comparison_report.get("comparisons", []):
            status = comparison_status(
                float(item["change_percent"]),
                noise_percent,
                float(comparison_report["maximum_regression_percent"]),
            )
            counts[status] += 1
        slower_count = counts["PIU' LENTO"]
        print(
            "Confronto baseline: "
            f'{counts["MIGLIORATO"]} migliorati | {counts["STABILE"]} stabili | '
            f"{slower_count} piu' lenti | {counts['REGRESSIONE']} regressioni | "
            f'{len(comparison_report.get("missing", []))} mancanti'
        )

    print_matmul_summary(results)
    if output_path is not None:
        print(f"\nRisultati completi: {output_path}")


def main() -> int:
    arguments = parse_arguments()
    if arguments.max_regression_percent < 0.0 or arguments.noise_percent < 0.0:
        print("comparison percentages must be non-negative", file=sys.stderr)
        return 2
    if not arguments.benchmark.is_file():
        print(
            f"benchmark executable not found: {arguments.benchmark}\n"
            "Build it with: cmake --build --preset release --target runtime_benchmark",
            file=sys.stderr,
        )
        return 2

    scenarios = SMOKE_SCENARIOS if arguments.smoke else SCENARIOS
    started = time.monotonic()
    comparison_report = None
    print("Suite prestazionale rappresentativa CPU/Metal")
    print("La barra avanza quando un singolo test viene realmente completato.")
    try:
        records = run_scenarios(arguments.benchmark, scenarios, arguments.smoke)
        if arguments.output is not None:
            write_jsonl(arguments.output, records)

        if arguments.baseline is not None:
            baseline = load_results(arguments.baseline)
            current = current_results_map(records)
            comparison_report = compare(
                baseline, current, arguments.max_regression_percent
            )
            print_comparison(comparison_report, arguments.noise_percent)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    elapsed = time.monotonic() - started
    print_final_summary(
        records, elapsed, comparison_report, arguments.noise_percent, arguments.output
    )
    if comparison_report is None:
        return 0
    return 1 if comparison_report["regressions"] or comparison_report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
