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
    if len(sys.argv) != 5:
        raise SystemExit("uso: test_dataset_cli.py CLI DOCUMENTS CORPUS_A CORPUS_B")

    cli = Path(sys.argv[1])
    documents = Path(sys.argv[2])
    corpus_inputs = [Path(sys.argv[3]), Path(sys.argv[4])]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = root / "tiny.llmtok"
        prefix = root / "tiny-dataset"
        run([str(cli), "tokenizer", "train", str(model), "264", *map(str, corpus_inputs)])

        prepared = run(
            [str(cli), "dataset", "prepare", str(model), str(documents), str(prefix)]
        )
        if "Dataset:" not in prepared.stderr or "100.0%" not in prepared.stderr:
            raise AssertionError(f"avanzamento dataset assente:\n{prepared.stderr}")
        report = json.loads(prepared.stdout)
        if report["schema"] != "llm-lab-dataset-report-v1":
            raise AssertionError(report)
        if report["tokenizer_vocabulary_size"] != 264:
            raise AssertionError(report)
        if report["model_vocabulary_size"] != 265:
            raise AssertionError(report)
        if report["end_of_document_token"] != 264:
            raise AssertionError(report)
        for split in ("train", "validation", "test"):
            if report["splits"][split]["documents"] != 2:
                raise AssertionError(report)
            if report["splits"][split]["tokens"] < 2:
                raise AssertionError(report)
            path = Path(f"{prefix}.{split}.llmdat")
            if not path.is_file() or path.stat().st_size <= 128:
                raise AssertionError(f"artefatto assente o vuoto: {path}")

        repeated = subprocess.run(
            [str(cli), "dataset", "prepare", str(model), str(documents), str(prefix)],
            check=False,
            capture_output=True,
            text=True,
        )
        if repeated.returncode == 0 or "already exists" not in repeated.stderr:
            raise AssertionError("gli artefatti esistenti dovrebbero essere rifiutati")

        malformed = root / "malformed.jsonl"
        malformed.write_text('{"id":"broken","text":}\n', encoding="utf-8")
        malformed_prefix = root / "malformed-dataset"
        invalid = subprocess.run(
            [str(cli), "dataset", "prepare", str(model), str(malformed), str(malformed_prefix)],
            check=False,
            capture_output=True,
            text=True,
        )
        if invalid.returncode == 0 or "invalid documents JSONL" not in invalid.stderr:
            raise AssertionError("il JSONL malformato dovrebbe essere rifiutato")
        if list(root.glob("malformed-dataset*.llmdat")) or list(
            root.glob("malformed-dataset*.part")
        ):
            raise AssertionError("un errore non deve pubblicare file parziali")

        duplicate = root / "duplicate.jsonl"
        duplicate.write_text(
            '{"id":"doc-0","text":"primo"}\n'
            '{"id":"doc-0","text":"secondo"}\n',
            encoding="utf-8",
        )
        duplicate_prefix = root / "duplicate-dataset"
        invalid = subprocess.run(
            [str(cli), "dataset", "prepare", str(model), str(duplicate), str(duplicate_prefix)],
            check=False,
            capture_output=True,
            text=True,
        )
        if invalid.returncode == 0 or "duplicate document ID" not in invalid.stderr:
            raise AssertionError("gli ID duplicati dovrebbero essere rifiutati")
        if list(root.glob("duplicate-dataset*.llmdat")) or list(
            root.glob("duplicate-dataset*.part")
        ):
            raise AssertionError("un duplicato non deve pubblicare file parziali")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
