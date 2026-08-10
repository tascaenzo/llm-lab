import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = PROJECT_ROOT / "utils" / "benchmarks" / "compare_results.py"


def result(median_seconds: float, threads: int = 4) -> dict:
    return {
        "type": "result",
        "operation": "matmul",
        "actual_threads": threads,
        "dimensions": {"rows": 128, "inner": 128, "columns": 128},
        "median_seconds": median_seconds,
    }


class BenchmarkComparisonTest(unittest.TestCase):
    def run_comparison(self, baseline_records, current_records, *extra_arguments):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jsonl"
            current = root / "current.jsonl"
            baseline.write_text(
                "\n".join(json.dumps(record) for record in baseline_records) + "\n",
                encoding="utf-8",
            )
            current.write_text(
                "\n".join(json.dumps(record) for record in current_records) + "\n",
                encoding="utf-8",
            )
            return subprocess.run(
                [
                    sys.executable,
                    str(COMPARE_SCRIPT),
                    str(baseline),
                    str(current),
                    *extra_arguments,
                ],
                check=False,
                capture_output=True,
                text=True,
            )

    def test_accepts_improvement(self):
        completed = self.run_comparison([result(1.0)], [result(0.8)])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["regressions"], [])

    def test_rejects_regression_over_threshold(self):
        completed = self.run_comparison(
            [result(1.0)],
            [result(1.2)],
            "--max-regression-percent",
            "10",
        )
        self.assertEqual(completed.returncode, 1)
        report = json.loads(completed.stdout)
        self.assertEqual(len(report["regressions"]), 1)

    def test_missing_result_can_be_allowed(self):
        completed = self.run_comparison(
            [result(1.0), result(0.5, threads=8)],
            [result(1.0)],
            "--allow-missing",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(len(report["missing"]), 1)


if __name__ == "__main__":
    unittest.main()
