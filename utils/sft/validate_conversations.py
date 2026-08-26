#!/usr/bin/env python3
"""Valida un corpus JSONL conversazionale prima della tokenizzazione SFT."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1


def fnv1a_64(value: str) -> int:
    result = FNV_OFFSET
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * FNV_PRIME) & UINT64_MASK
    return result


def split_for(identifier: str) -> str:
    bucket = fnv1a_64(identifier) % 10000
    if bucket < 9000:
        return "train"
    if bucket < 9500:
        return "validation"
    return "test"


def nonempty_text(record: dict, key: str, line_number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"riga {line_number}: '{key}' deve essere una stringa non vuota")
    if "\x00" in value:
        raise ValueError(f"riga {line_number}: '{key}' contiene NUL")
    return value


def validate_messages(value: object, line_number: int) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"riga {line_number}: 'messages' deve contenere almeno user/assistant")
    expected = "user"
    start = 0
    if isinstance(value[0], dict) and value[0].get("role") == "system":
        nonempty_text(value[0], "content", line_number)
        start = 1
    if start == len(value):
        raise ValueError(f"riga {line_number}: manca il dialogo dopo system")
    assistant_turns = 0
    characters = 0
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"riga {line_number}: messaggio {index} non e' un oggetto")
        role = message.get("role")
        content = nonempty_text(message, "content", line_number)
        characters += len(content)
        if index < start:
            continue
        if role != expected:
            raise ValueError(
                f"riga {line_number}: ruolo {index} '{role}', atteso '{expected}'"
            )
        if role == "assistant":
            assistant_turns += 1
            expected = "user"
        else:
            expected = "assistant"
    if expected != "user":
        raise ValueError(f"riga {line_number}: la conversazione deve terminare con assistant")
    return assistant_turns, characters


def validate(path: Path, maximum_characters: int | None) -> dict:
    identifiers: set[str] = set()
    splits = collections.Counter()
    sources = collections.Counter()
    licenses = collections.Counter()
    examples = 0
    messages = 0
    assistant_turns = 0
    characters = 0
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"riga {line_number}: le righe vuote non sono ammesse")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"riga {line_number}: JSON non valido: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"riga {line_number}: la radice deve essere un oggetto")
            identifier = nonempty_text(record, "id", line_number)
            source_name = nonempty_text(record, "source", line_number)
            license_name = nonempty_text(record, "license", line_number)
            if identifier in identifiers:
                raise ValueError(f"riga {line_number}: id duplicato '{identifier}'")
            identifiers.add(identifier)
            turns, example_characters = validate_messages(record.get("messages"), line_number)
            if maximum_characters is not None and example_characters > maximum_characters:
                raise ValueError(
                    f"riga {line_number}: {example_characters} caratteri superano --max-characters"
                )
            examples += 1
            messages += len(record["messages"])
            assistant_turns += turns
            characters += example_characters
            splits[split_for(identifier)] += 1
            sources[source_name] += 1
            licenses[license_name] += 1
    if examples == 0:
        raise ValueError("il corpus e' vuoto")
    missing_splits = [name for name in ("train", "validation", "test") if splits[name] == 0]
    if missing_splits:
        raise ValueError("split vuoti dopo hashing degli id: " + ", ".join(missing_splits))
    return {
        "schema": "llm-lab-sft-validation-v1",
        "valid": True,
        "examples": examples,
        "messages": messages,
        "assistant_turns": assistant_turns,
        "characters": characters,
        "splits": dict(sorted(splits.items())),
        "sources": dict(sorted(sources.items())),
        "licenses": dict(sorted(licenses.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="corpus conversations.jsonl")
    parser.add_argument(
        "--max-characters",
        type=int,
        help="limite preliminare; il limite autorevole resta quello tokenizzato della CLI",
    )
    parser.add_argument("--report", type=Path, help="scrive anche il report JSON in questo file")
    arguments = parser.parse_args()
    if arguments.max_characters is not None and arguments.max_characters < 1:
        parser.error("--max-characters deve essere positivo")
    try:
        report = validate(arguments.input, arguments.max_characters)
    except (OSError, ValueError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)
    print(encoded)
    if arguments.report is not None:
        arguments.report.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
