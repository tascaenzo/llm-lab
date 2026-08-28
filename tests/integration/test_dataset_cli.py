#!/usr/bin/env python3

import json
import hashlib
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def run(
    command,
    expected_returncode=0,
    *,
    cwd=None,
    environment=None,
    use_default_backend=True,
):
    active_environment = os.environ.copy()
    if use_default_backend:
        active_environment["LLM_LAB_BACKEND"] = "cpu"
    else:
        active_environment.pop("LLM_LAB_BACKEND", None)
    if environment is not None:
        active_environment.update(environment)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=active_environment,
    )
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


def fnv1a_bucket(text):
    value = 14695981039346656037
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value % 10000


def write_sft_examples(path):
    examples = []
    found = set()
    for number in range(10000):
        identifier = f"integration-sft:{number}"
        bucket = fnv1a_bucket(identifier)
        split = "train" if bucket < 9000 else "validation" if bucket < 9500 else "test"
        if split in found:
            continue
        found.add(split)
        examples.append(
            {
                "id": identifier,
                "source": "integration-fixture",
                "license": "CC0-1.0",
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ],
            }
        )
        if len(found) == 3:
            break
    path.write_text("".join(json.dumps(item) + "\n" for item in examples), encoding="utf-8")


def main():
    if len(sys.argv) != 5:
        raise SystemExit("uso: test_dataset_cli.py CLI DOCUMENTS CORPUS_A CORPUS_B")

    cli = Path(sys.argv[1])
    documents = Path(sys.argv[2])
    corpus_inputs = [Path(sys.argv[3]), Path(sys.argv[4])]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = root / "tiny.llmtok"
        wrong_model = root / "wrong-tiny.llmtok"
        prefix = root / "tiny-dataset"
        run([str(cli), "tokenizer", "train", str(model), "264", *map(str, corpus_inputs)])
        run([str(cli), "tokenizer", "train", str(wrong_model), "264", str(corpus_inputs[1])])

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
        corrupted_checkpoint = root / "generation-corrupted.llmckpt"
        corrupted_bytes = bytearray(generation_checkpoint.read_bytes())
        corrupted_bytes[-1] ^= 1
        corrupted_checkpoint.write_bytes(corrupted_bytes)
        corrupted = run(
            [str(cli), "model", "generate", str(corrupted_checkpoint), str(model), "1", "ciao"],
            expected_returncode=1,
        )
        if "Loading checkpoint failed" not in corrupted.stderr:
            raise AssertionError(f"checkpoint corrotto accettato:\n{corrupted.stderr}")
        scalable_checkpoint = root / "scalable.llmckpt"
        scalable_best_checkpoint = root / "scalable-best.llmckpt"
        scalable_log = root / "scalable.jsonl"
        scalable = json.loads(
            run(
                [
                    str(cli),
                    "model",
                    "train",
                    f"{prefix}.train.llmdat",
                    "2",
                    "--batch-size",
                    "1",
                    "--context",
                    "2",
                    "--hidden",
                    "4",
                    "--layers",
                    "2",
                    "--heads",
                    "2",
                    "--ffn",
                    "8",
                    "--learning-rate",
                    "0.001",
                    "--seed",
                    "123",
                    "--checkpoint",
                    str(scalable_checkpoint),
                    "--validation",
                    f"{prefix}.validation.llmdat",
                    "--validation-every",
                    "1",
                    "--validation-batches",
                    "1",
                    "--best-checkpoint",
                    str(scalable_best_checkpoint),
                    "--log",
                    str(scalable_log),
                ]
            ).stdout
        )
        if (
            scalable["layer_count"] != 2
            or scalable["head_count"] != 2
            or scalable["feed_forward_size"] != 8
            or scalable["parameter_count"] <= training_report["parameter_count"]
            or not scalable_checkpoint.is_file()
            or not scalable_best_checkpoint.is_file()
            or not Path(f"{scalable_best_checkpoint}.metrics.json").is_file()
            or not math.isfinite(scalable["validation_loss"])
        ):
            raise AssertionError(scalable)
        log_events = [json.loads(line) for line in scalable_log.read_text().splitlines()]
        if (
            [event["event"] for event in log_events].count("train") != 2
            or [event["event"] for event in log_events].count("validation") != 2
            or log_events[-1]["event"] != "validation"
            or not math.isfinite(log_events[-1]["best_loss"])
            or not any(event.get("improved") for event in log_events)
        ):
            raise AssertionError(log_events)
        scalable_evaluation = json.loads(
            run(
                [
                    str(cli),
                    "model",
                    "evaluate",
                    f"{prefix}.validation.llmdat",
                    str(scalable_checkpoint),
                    "1",
                    "--batch-size",
                    "1",
                    "--seed",
                    "123",
                ]
            ).stdout
        )
        if not math.isfinite(scalable_evaluation["loss"]):
            raise AssertionError(scalable_evaluation)
        scalable_overfit_command = [
            str(cli),
            "model",
            "train",
            f"{prefix}.train.llmdat",
            "1",
            "--batch-size",
            "1",
            "--context",
            "2",
            "--hidden",
            "8",
            "--layers",
            "2",
            "--heads",
            "2",
            "--ffn",
            "16",
            "--learning-rate",
            "0.01",
            "--seed",
            "77",
        ]
        scalable_initial_loss = json.loads(run(scalable_overfit_command).stdout)["loss"]
        scalable_overfit_command[4] = "200"
        scalable_final_loss = json.loads(run(scalable_overfit_command).stdout)["loss"]
        if not scalable_final_loss < scalable_initial_loss * 0.8:
            raise AssertionError((scalable_initial_loss, scalable_final_loss))
        generated = run(
            [str(cli), "model", "generate", str(generation_checkpoint), str(model), "2", "ciao"]
        )
        if not generated.stdout.strip():
            raise AssertionError("generazione vuota")

        # The base pretraining vocabulary can reserve IDs for a future SFT
        # protocol.  Generation must keep those IDs out until they have a
        # textual definition, rather than rejecting the checkpoint or trying
        # to decode an unknown token.
        reserved_prefix = root / "reserved-dataset"
        reserved_report = json.loads(
            run(
                [
                    str(cli),
                    "dataset",
                    "prepare",
                    str(model),
                    str(documents),
                    str(reserved_prefix),
                    "--reserved-tokens",
                    "3",
                ]
            ).stdout
        )
        if reserved_report["model_vocabulary_size"] != 268:
            raise AssertionError(reserved_report)
        reserved_checkpoint = root / "reserved.llmckpt"
        run(
            [
                str(cli),
                "model",
                "train",
                f"{reserved_prefix}.train.llmdat",
                "2",
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
                str(reserved_checkpoint),
            ]
        )
        reserved_generated = run(
            [str(cli), "model", "generate", str(reserved_checkpoint), str(model), "2", "ciao"]
        )
        if not reserved_generated.stdout.strip():
            raise AssertionError("generazione con ID riservati vuota")

        sft_base_prefix = root / "sft-base-dataset"
        sft_base_report = json.loads(
            run(
                [
                    str(cli),
                    "dataset",
                    "prepare",
                    str(model),
                    str(documents),
                    str(sft_base_prefix),
                    "--reserved-tokens",
                    "7",
                ]
            ).stdout
        )
        if sft_base_report["model_vocabulary_size"] != 272:
            raise AssertionError(sft_base_report)
        sft_base_checkpoint = root / "sft-base.llmckpt"
        run(
            [
                str(cli),
                "model",
                "train",
                f"{sft_base_prefix}.train.llmdat",
                "1",
                "--batch-size",
                "1",
                "--context",
                "16",
                "--hidden",
                "4",
                "--learning-rate",
                "0.001",
                "--checkpoint",
                str(sft_base_checkpoint),
            ]
        )
        sft_jsonl = root / "conversations.jsonl"
        write_sft_examples(sft_jsonl)
        sft_prefix = root / "italiano-chat"
        sft_dataset_report = json.loads(
            run(
                [
                    str(cli),
                    "dataset",
                    "sft-prepare",
                    str(model),
                    str(sft_jsonl),
                    str(sft_prefix),
                    "--context",
                    "16",
                ]
            ).stdout
        )
        if (
            sft_dataset_report["schema"] != "llm-lab-sft-dataset-report-v1"
            or sft_dataset_report["protocol"]["assistant"] != 267
            or any(sft_dataset_report["splits"][split]["examples"] != 1 for split in ("train", "validation", "test"))
        ):
            raise AssertionError(sft_dataset_report)
        sft_checkpoint = root / "italiano-chat.llmckpt"
        sft_best = root / "italiano-chat-best.llmckpt"
        sft_log = root / "italiano-chat.jsonl"
        sft_training = json.loads(
            run(
                [
                    str(cli),
                    "model",
                    "sft",
                    f"{sft_prefix}.train.llmsft",
                    "2",
                    "--base",
                    str(sft_base_checkpoint),
                    "--batch-size",
                    "1",
                    "--learning-rate",
                    "0.001",
                    "--min-learning-rate",
                    "0.0001",
                    "--checkpoint",
                    str(sft_checkpoint),
                    "--validation",
                    f"{sft_prefix}.validation.llmsft",
                    "--validation-every",
                    "1",
                    "--validation-batches",
                    "1",
                    "--best-checkpoint",
                    str(sft_best),
                    "--log",
                    str(sft_log),
                ]
            ).stdout
        )
        if (
            sft_training["schema"] != "llm-lab-sft-training-v1"
            or sft_training["steps"] != 2
            or not math.isfinite(sft_training["assistant_loss"])
            or not sft_checkpoint.is_file()
            or not sft_best.is_file()
        ):
            raise AssertionError(sft_training)
        run(
            [
                str(cli),
                "model",
                "chat",
                str(sft_checkpoint),
                str(model),
                "2",
                "a",
                "--system",
                "s",
                "--backend",
                "cpu",
            ]
        )
        chat_without_system = run(
            [
                str(cli),
                "model",
                "chat",
                str(sft_checkpoint),
                str(model),
                "2",
                "a",
                "--backend",
                "cpu",
            ]
        )
        if "prefill 4 token" not in chat_without_system.stderr:
            raise AssertionError(
                "la chat senza --system deve serializzare soltanto user/end/assistant: "
                + chat_without_system.stderr
            )
        sampled_command = [
            str(cli),
            "model",
            "generate",
            str(generation_checkpoint),
            str(model),
            "4",
            "L'Italia è",
            "--temperature",
            "0.8",
            "--top-k",
            "8",
            "--repetition-penalty",
            "1.1",
            "--seed",
            "73",
            "--backend",
            "cpu",
        ]
        sampled_first = run(sampled_command)
        sampled_second = run(sampled_command)
        if sampled_first.stdout != sampled_second.stdout:
            raise AssertionError("lo stesso seed deve produrre la stessa generazione")
        if "L'Italia è" not in sampled_first.stdout or "\\xc3\\xa8" in sampled_first.stdout:
            raise AssertionError(f"output UTF-8 inatteso: {sampled_first.stdout!r}")
        invalid_sampling = run(
            sampled_command[:-2] + ["--repetition-penalty", "0.9"], expected_returncode=1
        )
        if "Invalid model generation option" not in invalid_sampling.stderr:
            raise AssertionError("una repetition penalty inferiore a 1 deve essere rifiutata")
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
        if evaluated["backend"] != "cpu" or not evaluated["device"]:
            raise AssertionError(evaluated)

        diagnosed_result = run(
            [
                str(cli),
                "model",
                "diagnose",
                f"{prefix}.validation.llmdat",
                str(scalable_checkpoint),
                str(model),
                "1",
                "--batch-size",
                "1",
                "--seed",
                "123",
            ]
        )
        diagnosed = json.loads(diagnosed_result.stdout)
        if (
            diagnosed["schema"] != "llm-lab-model-diagnostics-v1"
            or diagnosed["tokens"] != 2
            or diagnosed["regular_tokens"] > diagnosed["tokens"]
            or not math.isfinite(diagnosed["mean_rank"])
            or not 0.0 <= diagnosed["top_1_accuracy"] <= diagnosed["top_5_accuracy"] <= 1.0
            or not diagnosed["top_5_accuracy"] <= diagnosed["top_20_accuracy"] <= 1.0
            or not diagnosed["top_20_accuracy"] <= diagnosed["top_100_accuracy"] <= 1.0
            or diagnosed["backend"] != "cpu"
        ):
            raise AssertionError(diagnosed)
        if "Sample reale, prompt:" not in diagnosed_result.stderr:
            raise AssertionError(diagnosed_result.stderr)
        wrong_diagnostic = run(
            [
                str(cli),
                "model",
                "diagnose",
                f"{prefix}.validation.llmdat",
                str(scalable_checkpoint),
                str(wrong_model),
                "1",
            ],
            expected_returncode=1,
        )
        if "does not match" not in wrong_diagnostic.stderr:
            raise AssertionError(wrong_diagnostic.stderr)

        environment_root = root / "environment"
        environment_root.mkdir()
        environment_file = environment_root / ".env"
        environment_file.write_text("LLM_LAB_BACKEND=cpu\n", encoding="utf-8")
        evaluation_command = [
            str(cli),
            "model",
            "evaluate",
            f"{prefix}.validation.llmdat",
            str(generation_checkpoint),
            "1",
            "--batch-size",
            "1",
        ]
        from_dotenv = json.loads(
            run(
                evaluation_command,
                cwd=environment_root,
                use_default_backend=False,
            ).stdout
        )
        if from_dotenv["backend"] != "cpu":
            raise AssertionError(from_dotenv)

        environment_file.write_text("LLM_LAB_BACKEND=invalid\n", encoding="utf-8")
        invalid_environment = run(
            evaluation_command,
            expected_returncode=1,
            cwd=environment_root,
            use_default_backend=False,
        )
        if "Invalid LLM_LAB_BACKEND" not in invalid_environment.stderr:
            raise AssertionError(invalid_environment.stderr)
        forced_cpu = json.loads(
            run(
                [*evaluation_command, "--backend", "cpu"],
                cwd=environment_root,
                use_default_backend=False,
            ).stdout
        )
        if forced_cpu["backend"] != "cpu":
            raise AssertionError(forced_cpu)
        exported_cpu = json.loads(
            run(
                evaluation_command,
                cwd=environment_root,
                environment={"LLM_LAB_BACKEND": "cpu"},
                use_default_backend=False,
            ).stdout
        )
        if exported_cpu["backend"] != "cpu":
            raise AssertionError(exported_cpu)

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

        equivalent_source = root / "equivalent-source.llmckpt"
        run(
            [
                str(cli),
                "model",
                "train",
                str(overfit_dataset),
                "1",
                "--batch-size",
                "1",
                "--gradient-accumulation",
                "2",
                "--context",
                "1",
                "--hidden",
                "4",
                "--learning-rate",
                "0.05",
                "--seed",
                "19",
                "--checkpoint",
                str(equivalent_source),
            ]
        )
        equivalent_log = root / "equivalent-resume.jsonl"
        equivalent_result = json.loads(
            run(
                [
                    str(cli),
                    "model",
                    "train",
                    str(overfit_dataset),
                    "1",
                    "--resume",
                    str(equivalent_source),
                    "--batch-size",
                    "2",
                    "--gradient-accumulation",
                    "1",
                    "--checkpoint",
                    str(root / "equivalent-candidate.llmckpt"),
                    "--log",
                    str(equivalent_log),
                ]
            ).stdout
        )
        equivalent_run = json.loads(equivalent_log.read_text().splitlines()[0])
        if (
            equivalent_result["steps"] != 2
            or equivalent_result["gradient_accumulation_steps"] != 1
            or equivalent_run["batch_size"] != 2
            or equivalent_run["gradient_accumulation"] != 1
        ):
            raise AssertionError((equivalent_result, equivalent_run))
        invalid_override = run(
            [
                str(cli),
                "model",
                "train",
                str(overfit_dataset),
                "1",
                "--resume",
                str(equivalent_source),
                "--batch-size",
                "3",
                "--gradient-accumulation",
                "1",
            ],
            expected_returncode=1,
        )
        if "invalid argument" not in invalid_override.stderr.lower():
            raise AssertionError(invalid_override.stderr)

        scheduled_checkpoint = root / "scheduled.llmckpt"
        scheduled_log = root / "scheduled-training.jsonl"
        scheduled_result = run(
            [
                str(cli),
                "model",
                "train",
                str(overfit_dataset),
                "3",
                "--batch-size",
                "1",
                "--context",
                "1",
                "--hidden",
                "4",
                "--learning-rate",
                "0.05",
                "--min-learning-rate",
                "0.01",
                "--gradient-accumulation",
                "2",
                "--warmup-steps",
                "2",
                "--total-steps",
                "4",
                "--gradient-clip",
                "1",
                "--seed",
                "19",
                "--checkpoint",
                str(scheduled_checkpoint),
                "--checkpoint-every",
                "1",
                "--log",
                str(scheduled_log),
            ]
        )
        scheduled = json.loads(scheduled_result.stdout)
        if (
            scheduled["gradient_accumulation_steps"] != 2
            or not math.isclose(scheduled["learning_rate"], 0.03, rel_tol=0.0, abs_tol=1e-7)
            or not math.isfinite(scheduled["gradient_norm"])
            or scheduled["gradient_norm"] <= 0.0
            or scheduled["sampling"] != "shuffled"
            or not scheduled_checkpoint.is_file()
        ):
            raise AssertionError(scheduled)
        if 'totale 75.00% (step 3/4)' not in scheduled_result.stderr:
            raise AssertionError(f"avanzamento totale assente:\n{scheduled_result.stderr}")
        train_events = [
            json.loads(line)
            for line in scheduled_log.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("event") == "train"
        ]
        if len(train_events) != 3:
            raise AssertionError(f"eventi di training mancanti: {train_events}")
        telemetry = train_events[-1]
        required_telemetry = {
            "accelerator_backend",
            "accelerator_device",
            "accelerator_active_buffer_count",
            "accelerator_active_bytes",
            "accelerator_peak_active_bytes",
            "accelerator_cached_buffer_count",
            "accelerator_cached_bytes",
            "accelerator_dispatches",
            "accelerator_synchronizations",
            "accelerator_reused_buffer_allocations",
            "accelerator_total_gpu_seconds",
            "accelerator_last_gpu_seconds",
            "metal_active_bytes",
            "metal_peak_active_bytes",
            "metal_total_gpu_seconds",
        }
        if telemetry.get("accelerator_backend") != "cpu" or not required_telemetry <= telemetry.keys():
            raise AssertionError(f"telemetria acceleratore incompleta: {telemetry}")
        scheduled_resumed = json.loads(
            run(
                [
                    str(cli),
                    "model",
                    "train",
                    str(overfit_dataset),
                    "1",
                    "--resume",
                    str(scheduled_checkpoint),
                    "--checkpoint",
                    str(root / "scheduled-resumed.llmckpt"),
                ]
            ).stdout
        )
        if scheduled_resumed["steps"] != 4 or not math.isclose(
            scheduled_resumed["learning_rate"], 0.01, rel_tol=0.0, abs_tol=1e-7
        ):
            raise AssertionError(scheduled_resumed)

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
