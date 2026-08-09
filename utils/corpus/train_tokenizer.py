#!/usr/bin/env python3
"""Addestra un tokenizer da un corpus derivato e ne registra la provenienza."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_VOCABULARY_SIZE = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def model_merge_count(path: Path) -> int:
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 3 or lines[0] != "llm-tokenizer 1" or lines[1] != "base-vocab 256":
        raise RuntimeError("il trainer ha prodotto un file .llmtok non valido")
    prefix = "merges "
    if not lines[2].startswith(prefix):
        raise RuntimeError("il trainer ha prodotto un file .llmtok senza numero di merge")
    merge_count = int(lines[2][len(prefix) :])
    if merge_count < 0 or len(lines) != merge_count + 3:
        raise RuntimeError("il numero di merge nel file .llmtok non coincide con le righe")
    return merge_count


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_trainer = project_root / "build" / "debug" / (
        "llm-lab.exe" if sys.platform == "win32" else "llm-lab"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=project_root / "data" / "clean" / "italiano-wikipedia-v1" / "manifest.json",
    )
    parser.add_argument("--trainer", type=Path, default=default_trainer)
    parser.add_argument("--output", type=Path, help="file .llmtok da creare")
    parser.add_argument("--vocab-size", type=int, default=32000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    try:
        if args.vocab_size < BASE_VOCABULARY_SIZE:
            raise RuntimeError("--vocab-size deve essere almeno 256")
        if not args.trainer.is_file():
            raise RuntimeError(f"eseguibile trainer non trovato: {args.trainer}")
        if not args.corpus_manifest.is_file():
            raise RuntimeError(f"manifest del corpus non trovato: {args.corpus_manifest}")

        corpus_manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
        corpus_name = corpus_manifest.get("name")
        tokenizer_inputs = corpus_manifest.get("outputs", {}).get("tokenizer_input")
        if not isinstance(corpus_name, str) or not isinstance(tokenizer_inputs, list) or not tokenizer_inputs:
            raise RuntimeError("manifest del corpus incompleto")

        input_paths = [project_root / path for path in tokenizer_inputs]
        if any(not path.is_file() for path in input_paths):
            raise RuntimeError("una parte del corpus derivato non esiste")

        output_path = args.output
        if output_path is None:
            output_path = project_root / "artifacts" / "tokenizers" / f"{corpus_name}.llmtok"
        output_path = output_path.resolve()
        metadata_path = output_path.with_suffix(output_path.suffix + ".json")
        temporary_output_path = output_path.with_suffix(output_path.suffix + ".part")
        if output_path.exists() or metadata_path.exists() or temporary_output_path.exists():
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
        subprocess.run(command, check=True)

        merge_count = model_merge_count(temporary_output_path)
        temporary_output_path.replace(output_path)
        artifact_manifest = {
            "schema": "llm-lab-tokenizer-artifact-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tokenizer": {
                "file": project_relative(output_path, project_root),
                "sha256": sha256_file(output_path),
                "target_vocabulary_size": args.vocab_size,
                "actual_vocabulary_size": BASE_VOCABULARY_SIZE + merge_count,
                "merge_count": merge_count,
            },
            "trainer": project_relative(args.trainer, project_root),
            "corpus": {
                "manifest": project_relative(args.corpus_manifest, project_root),
                "manifest_sha256": sha256_file(args.corpus_manifest),
                "tokenizer_input": [project_relative(path, project_root) for path in input_paths],
            },
            "command": command,
        }
        metadata_path.write_text(
            json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Tokenizer creato: {output_path}")
        print(f"Vocabolario effettivo: {BASE_VOCABULARY_SIZE + merge_count}")
        print(f"SHA-256: {artifact_manifest['tokenizer']['sha256']}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
