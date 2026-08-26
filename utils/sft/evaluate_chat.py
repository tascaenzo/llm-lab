#!/usr/bin/env python3
"""Esegue una suite chat riproducibile e salva risposte e metadati in JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_suite(path: Path) -> list[dict]:
    records = []
    identifiers = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"suite, riga {line_number}: JSON non valido") from error
            if not isinstance(record, dict):
                raise ValueError(f"suite, riga {line_number}: oggetto atteso")
            identifier = record.get("id")
            prompt = record.get("prompt")
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise ValueError(f"suite, riga {line_number}: id assente o duplicato")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"suite, riga {line_number}: prompt assente")
            identifiers.add(identifier)
            records.append(record)
    if not records:
        raise ValueError("suite vuota")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error(f"l'output esiste gia': {arguments.output}")
    if arguments.tokens < 1 or arguments.top_k < 1:
        parser.error("--tokens e --top-k devono essere positivi")
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".part")
    try:
        suite = load_suite(arguments.suite)
        metadata = {
            "schema": "llm-lab-chat-evaluation-run-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": sha256(arguments.checkpoint),
            "tokenizer_sha256": sha256(arguments.tokenizer),
            "suite_sha256": sha256(arguments.suite),
            "backend": arguments.backend,
            "tokens": arguments.tokens,
            "temperature": arguments.temperature,
            "top_k": arguments.top_k,
            "repetition_penalty": arguments.repetition_penalty,
            "seed": arguments.seed,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for index, record in enumerate(suite, start=1):
                command = [
                    str(arguments.executable), "model", "chat", str(arguments.checkpoint),
                    str(arguments.tokenizer), str(arguments.tokens), record["prompt"],
                    "--temperature", str(arguments.temperature), "--top-k", str(arguments.top_k),
                    "--repetition-penalty", str(arguments.repetition_penalty),
                    "--seed", str(arguments.seed), "--backend", arguments.backend,
                ]
                if isinstance(record.get("system"), str):
                    command.extend(("--system", record["system"]))
                completed = subprocess.run(command, text=True, capture_output=True, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"generazione {record['id']} fallita: {completed.stderr.strip()}"
                    )
                result = {
                    "schema": "llm-lab-chat-evaluation-result-v1",
                    "id": record["id"],
                    "category": record.get("category"),
                    "prompt": record["prompt"],
                    "criteria": record.get("criteria", []),
                    "answer": completed.stdout.rstrip("\n"),
                }
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"[{index}/{len(suite)}] {record['id']}", file=sys.stderr)
        temporary.replace(arguments.output)
    except (OSError, ValueError, RuntimeError) as error:
        temporary.unlink(missing_ok=True)
        print(f"Errore: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
