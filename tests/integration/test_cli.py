#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command, expected_returncode=0):
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != expected_returncode:
        raise AssertionError(
            f"comando {command!r}: atteso {expected_returncode}, ottenuto {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main():
    if len(sys.argv) != 4:
        raise SystemExit("uso: test_cli.py CLI CORPUS_A CORPUS_B")

    cli = Path(sys.argv[1])
    inputs = [Path(sys.argv[2]), Path(sys.argv[3])]
    expected_bytes = sum(path.stat().st_size for path in inputs)

    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        model = temporary_root / "tiny.llmtok"
        trained = run([str(cli), "tokenizer", "train", str(model), "264", *map(str, inputs)])
        if not model.is_file() or "with 264 tokens" not in trained.stdout:
            raise AssertionError(f"output di training inatteso: {trained.stdout!r}")

        evaluated = run(
            [str(cli), "tokenizer", "evaluate", str(model), "1048576", *map(str, inputs)]
        )
        report = json.loads(evaluated.stdout)
        if report["schema"] != "llm-lab-tokenizer-evaluation-v1":
            raise AssertionError(report)
        if report["bytes"] != expected_bytes or report["tokens"] <= 0:
            raise AssertionError(report)
        if report["round_trip"] is not True or report["bytes_per_token"] <= 0:
            raise AssertionError(report)

        invalid_limit = subprocess.run(
            [str(cli), "tokenizer", "evaluate", str(model), "0", str(inputs[0])],
            check=False,
            capture_output=True,
            text=True,
        )
        if invalid_limit.returncode == 0 or "MAX_BYTES" not in invalid_limit.stderr:
            raise AssertionError("il limite nullo dovrebbe essere rifiutato")

        malformed = temporary_root / "malformed.llmtok"
        malformed.write_text("not a tokenizer\n", encoding="ascii")
        invalid_model = subprocess.run(
            [str(cli), "tokenizer", "evaluate", str(malformed), "10", str(inputs[0])],
            check=False,
            capture_output=True,
            text=True,
        )
        if invalid_model.returncode == 0 or "invalid model" not in invalid_model.stderr.lower():
            raise AssertionError("un modello malformato dovrebbe essere rifiutato")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
