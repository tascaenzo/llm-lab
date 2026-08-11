#!/usr/bin/env python3
"""Compare JSONL runtime benchmark results produced on the same controlled host."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ResultKey = Tuple[str, str, str, str, int, str]


def result_key(record: Dict[str, Any]) -> ResultKey:
    return (
        str(record.get("scenario", "default")),
        str(record.get("backend", "cpu")),
        str(record.get("dtype", "f32")),
        str(record["operation"]),
        int(record["actual_threads"]),
        json.dumps(record["dimensions"], sort_keys=True, separators=(",", ":")),
    )


def load_results(path: Path) -> Dict[ResultKey, Dict[str, Any]]:
    results: Dict[ResultKey, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if record.get("type") != "result":
                continue
            key = result_key(record)
            if key in results:
                raise ValueError(f"{path}:{line_number}: duplicate benchmark result {key}")
            median = float(record["median_seconds"])
            if median <= 0.0:
                raise ValueError(f"{path}:{line_number}: median_seconds must be positive")
            results[key] = record
    if not results:
        raise ValueError(f"{path}: no benchmark result records")
    return results


def compare(
    baseline: Dict[ResultKey, Dict[str, Any]],
    current: Dict[ResultKey, Dict[str, Any]],
    maximum_regression_percent: float,
) -> Dict[str, Any]:
    regressions: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    compared = 0
    for key, baseline_record in baseline.items():
        current_record = current.get(key)
        if current_record is None:
            missing.append(
                {
                    "scenario": key[0],
                    "backend": key[1],
                    "dtype": key[2],
                    "operation": key[3],
                    "actual_threads": key[4],
                    "dimensions": json.loads(key[5]),
                }
            )
            continue
        compared += 1
        baseline_seconds = float(baseline_record["median_seconds"])
        current_seconds = float(current_record["median_seconds"])
        regression_percent = (current_seconds / baseline_seconds - 1.0) * 100.0
        comparison = {
            "scenario": key[0],
            "backend": key[1],
            "dtype": key[2],
            "operation": key[3],
            "actual_threads": key[4],
            "dimensions": json.loads(key[5]),
            "baseline_seconds": baseline_seconds,
            "current_seconds": current_seconds,
            "change_percent": regression_percent,
        }
        comparisons.append(comparison)
        if regression_percent > maximum_regression_percent:
            regressions.append({**comparison, "regression_percent": regression_percent})
    return {
        "type": "benchmark_comparison",
        "schema_version": 1,
        "maximum_regression_percent": maximum_regression_percent,
        "baseline_records": len(baseline),
        "current_records": len(current),
        "compared_records": compared,
        "comparisons": comparisons,
        "regressions": regressions,
        "missing": missing,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two llm-lab benchmark JSONL files from the same host."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument(
        "--max-regression-percent",
        type=float,
        default=10.0,
        help="maximum accepted median-time regression (default: 10)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="do not fail when a baseline case is absent from current results",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.max_regression_percent < 0.0:
        print("max regression percent must be non-negative", file=sys.stderr)
        return 2
    try:
        baseline = load_results(arguments.baseline)
        current = load_results(arguments.current)
        report = compare(baseline, current, arguments.max_regression_percent)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    has_missing_failure = bool(report["missing"]) and not arguments.allow_missing
    return 1 if report["regressions"] or has_missing_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
