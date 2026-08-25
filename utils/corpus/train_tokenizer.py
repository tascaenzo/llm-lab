#!/usr/bin/env python3
"""Addestra un tokenizer da un corpus derivato e ne registra la provenienza."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


BASE_VOCABULARY_SIZE = 256
DEFAULT_EVALUATION_BYTES = 1024 * 1024
MODEL_MAGIC = b"LLMTOK\r\n"
MODEL_FORMAT_VERSION = 1
MODEL_HEADER = struct.Struct("<8sIIIIQ32s")
MODEL_MERGE_SIZE = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path, project_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        portable_path = resolved_path.relative_to(project_root.resolve())
    except ValueError:
        portable_path = resolved_path
    return portable_path.as_posix()


def model_merge_count(path: Path) -> int:
    data = path.read_bytes()
    if len(data) < MODEL_HEADER.size:
        raise RuntimeError("il trainer ha prodotto un file .llmtok non valido")
    magic, version, header_size, base_vocabulary_size, merge_count, payload_size, checksum = (
        MODEL_HEADER.unpack_from(data)
    )
    expected_payload_size = merge_count * MODEL_MERGE_SIZE
    payload = data[MODEL_HEADER.size :]
    if (
        magic != MODEL_MAGIC
        or version != MODEL_FORMAT_VERSION
        or header_size != MODEL_HEADER.size
        or base_vocabulary_size != BASE_VOCABULARY_SIZE
        or payload_size != expected_payload_size
        or len(payload) != expected_payload_size
        or hashlib.sha256(payload).digest() != checksum
    ):
        raise RuntimeError("il trainer ha prodotto un file .llmtok non valido")
    return merge_count


def current_git_commit(project_root: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def git_worktree_is_dirty(project_root: Path) -> Optional[bool]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout) if result.returncode == 0 else None


def child_peak_memory_kib() -> Optional[int]:
    try:
        import resource
    except ImportError:
        return None
    maximum = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        maximum /= 1024
    return int(maximum)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_trainer = project_root / "build" / "debug" / "llm-lab"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=project_root / "data" / "clean" / "italiano-wikipedia-v1" / "manifest.json",
    )
    parser.add_argument("--trainer", type=Path, default=default_trainer)
    parser.add_argument("--output", type=Path, help="file .llmtok da creare")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--evaluation-bytes", type=int, default=DEFAULT_EVALUATION_BYTES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    try:
        if args.vocab_size < BASE_VOCABULARY_SIZE:
            raise RuntimeError("--vocab-size deve essere almeno 256")
        if args.evaluation_bytes < 1:
            raise RuntimeError("--evaluation-bytes deve essere positivo")
        if not args.trainer.is_file():
            raise RuntimeError(f"eseguibile trainer non trovato: {args.trainer}")
        if not args.corpus_manifest.is_file():
            raise RuntimeError(f"manifest del corpus non trovato: {args.corpus_manifest}")

        corpus_manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
        corpus_name = corpus_manifest.get("name")
        tokenizer_inputs = corpus_manifest.get("outputs", {}).get("tokenizer_input")
        tokenizer_input_split = corpus_manifest.get("outputs", {}).get("tokenizer_input_split")
        if not isinstance(corpus_name, str) or not isinstance(tokenizer_inputs, list) or not tokenizer_inputs:
            raise RuntimeError("manifest del corpus incompleto")
        if tokenizer_input_split != "train":
            raise RuntimeError("il tokenizer richiede input dichiarati come split train-only")

        input_paths = [project_root / path for path in tokenizer_inputs]
        if any(not path.is_file() for path in input_paths):
            raise RuntimeError("una parte del corpus derivato non esiste")

        output_path = args.output
        if output_path is None:
            output_path = project_root / "artifacts" / "tokenizers" / f"{corpus_name}.llmtok"
        output_path = output_path.resolve()
        metadata_path = output_path.with_suffix(output_path.suffix + ".json")
        temporary_output_path = output_path.with_suffix(output_path.suffix + ".part")
        temporary_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".part")
        if any(
            path.exists()
            for path in (output_path, metadata_path, temporary_output_path, temporary_metadata_path)
        ):
            raise RuntimeError("l'artefatto di destinazione esiste gia'; scegli --output diverso")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.trainer),
            "tokenizer",
            "train",
            str(temporary_output_path),
            str(args.vocab_size),
            *(str(path) for path in input_paths),
        ]
        training_started_at = time.monotonic()
        subprocess.run(command, check=True)
        training_seconds = time.monotonic() - training_started_at
        training_peak_memory_kib = child_peak_memory_kib()

        merge_count = model_merge_count(temporary_output_path)
        evaluation_command = [
            str(args.trainer),
            "tokenizer",
            "evaluate",
            str(temporary_output_path),
            str(args.evaluation_bytes),
            *(str(path) for path in input_paths),
        ]
        evaluation_result = subprocess.run(
            evaluation_command, check=True, capture_output=True, text=True
        )
        evaluation = json.loads(evaluation_result.stdout)
        recorded_command = [
            project_relative(args.trainer, project_root),
            "tokenizer",
            "train",
            project_relative(output_path, project_root),
            str(args.vocab_size),
            *(project_relative(path, project_root) for path in input_paths),
        ]
        artifact_manifest = {
            "schema": "llm-lab-tokenizer-artifact-v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tokenizer": {
                "file": project_relative(output_path, project_root),
                "sha256": sha256_file(temporary_output_path),
                "format": {
                    "kind": "binary",
                    "version": MODEL_FORMAT_VERSION,
                    "byte_order": "little-endian",
                    "payload_checksum": "sha256",
                },
                "target_vocabulary_size": args.vocab_size,
                "actual_vocabulary_size": BASE_VOCABULARY_SIZE + merge_count,
                "merge_count": merge_count,
            },
            "trainer": {
                "file": project_relative(args.trainer, project_root),
                "sha256": sha256_file(args.trainer),
                "git_commit": current_git_commit(project_root),
                "git_worktree_dirty": git_worktree_is_dirty(project_root),
                "build_configuration": args.trainer.parent.name,
            },
            "execution": {
                "training_seconds": training_seconds,
                "training_peak_memory_kib": training_peak_memory_kib,
            },
            "evaluation": evaluation,
            "corpus": {
                "manifest": project_relative(args.corpus_manifest, project_root),
                "manifest_sha256": sha256_file(args.corpus_manifest),
                "tokenizer_input": [project_relative(path, project_root) for path in input_paths],
                "tokenizer_input_split": tokenizer_input_split,
                "snapshot": corpus_manifest,
            },
            "command": recorded_command,
        }
        temporary_metadata_path.write_text(
            json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_output_path.replace(output_path)
        temporary_metadata_path.replace(metadata_path)
        print(f"Tokenizer creato: {output_path}")
        print(f"Vocabolario effettivo: {BASE_VOCABULARY_SIZE + merge_count}")
        print(f"SHA-256: {artifact_manifest['tokenizer']['sha256']}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
