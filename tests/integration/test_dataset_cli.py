#!/usr/bin/env python3

import json
import hashlib
import math
import struct
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


def write_tiny_training_dataset(path):
    tokens = [0, 1] * 64
    payload = struct.pack(f"<{len(tokens)}I", *tokens)
    header = struct.pack(
        "<8sIIIIIIQQQ32s32s8s",
        b"LLMDATA\n",
        1,
        128,
        2,
        3,
        2,
        0,
        len(tokens),
        1,
        len(payload),
        bytes(32),
        hashlib.sha256(payload).digest(),
        bytes(8),
    )
    path.write_bytes(header + payload)


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

        generation_checkpoint = root / "generation.llmckpt"
        trained = run(
            [
                str(cli),
                "model",
                "train",
                f"{prefix}.train.llmdat",
                "4",
                "--batch-size",
                "1",
                "--context",
                "1",
                "--hidden",
                "4",
                "--learning-rate",
                "0.01",
                "--seed",
                "123",
                "--checkpoint",
                str(generation_checkpoint),
            ]
        )
        training_report = json.loads(trained.stdout)
        if training_report["schema"] != "llm-lab-model-training-v1":
            raise AssertionError(training_report)
        if training_report["steps"] != 4 or not math.isfinite(training_report["loss"]):
            raise AssertionError(training_report)
        if training_report["vocabulary_size"] != 265:
            raise AssertionError(training_report)
        if "Verifica dataset:" not in trained.stderr or "Training modello:" not in trained.stderr:
            raise AssertionError(f"avanzamento model assente:\n{trained.stderr}")
        generated = run(
            [str(cli), "model", "generate", str(generation_checkpoint), str(model), "2", "ciao"]
        )
        if not generated.stdout.strip():
            raise AssertionError("generazione vuota")
        evaluated = json.loads(
            run(
                [
                    str(cli),
                    "model",
                    "evaluate",
                    f"{prefix}.validation.llmdat",
                    str(generation_checkpoint),
                    "2",
                    "--batch-size",
                    "1",
                    "--seed",
                    "123",
                ]
            ).stdout
        )
        if evaluated["schema"] != "llm-lab-model-evaluation-v1" or not math.isfinite(
            evaluated["loss"]
        ):
            raise AssertionError(evaluated)

        overfit_dataset = root / "overfit.train.llmdat"
        write_tiny_training_dataset(overfit_dataset)
        training_command = [
            str(cli),
            "model",
            "train",
            str(overfit_dataset),
            "1",
            "--batch-size",
            "1",
            "--context",
            "1",
            "--hidden",
            "4",
            "--learning-rate",
            "0.05",
            "--seed",
            "19",
        ]
        initial_loss = json.loads(run(training_command).stdout)["loss"]
        training_command[4] = "200"
        final_loss = json.loads(run(training_command).stdout)["loss"]
        if not final_loss < initial_loss * 0.5:
            raise AssertionError((initial_loss, final_loss))

        continuous_checkpoint = root / "continuous.llmckpt"
        continuous_command = [
            str(cli),
            "model",
            "train",
            str(overfit_dataset),
            "5",
            "--batch-size",
            "1",
            "--context",
            "1",
            "--hidden",
            "4",
            "--learning-rate",
            "0.05",
            "--seed",
            "19",
            "--checkpoint",
            str(continuous_checkpoint),
        ]
        continuous = json.loads(run(continuous_command).stdout)
        resume_checkpoint = root / "resume.llmckpt"
        split_command = continuous_command.copy()
        split_command[4] = "3"
        split_command[-1] = str(resume_checkpoint)
        run(split_command)
        resumed_checkpoint = root / "resumed.llmckpt"
        resumed = json.loads(
            run(
                [
                    str(cli),
                    "model",
                    "train",
                    str(overfit_dataset),
                    "2",
                    "--resume",
                    str(resume_checkpoint),
                    "--checkpoint",
                    str(resumed_checkpoint),
                ]
            ).stdout
        )
        if not continuous_checkpoint.is_file() or not resumed_checkpoint.is_file():
            raise AssertionError("checkpoint non scritto")
        if resumed["steps"] != 5 or not math.isclose(
            resumed["loss"], continuous["loss"], rel_tol=0.0, abs_tol=1e-7
        ):
            raise AssertionError((continuous, resumed))

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
