#!/usr/bin/env python3
"""Derive context-safe SFT records while retaining the relevant dialogue window.

The original corpus is never changed. The first assistant response keeps its
standalone user request; later responses retain the previous complete exchange
as context. Long fields are compacted deliberately and recorded in provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


WHITESPACE = re.compile(r"\s+")


def compact(text: str, maximum: int) -> tuple[str, bool]:
    """Normalize whitespace and shorten on a word boundary when needed."""
    value = WHITESPACE.sub(" ", text).strip()
    if len(value) <= maximum:
        return value, False
    boundary = value.rfind(" ", 0, maximum - 1)
    if boundary < maximum // 2:
        boundary = maximum - 1
    return value[:boundary].rstrip(" ,;:") + "…", True


def include_system(source_id: str) -> bool:
    """Keep a stable 20% sample of original system prompts."""
    return hashlib.sha256(source_id.encode("utf-8")).digest()[0] % 5 == 0


def compact_message(role: str, content: str, maximum: int) -> tuple[dict[str, str] | None, bool]:
    value, shortened = compact(content, maximum)
    return ({"role": role, "content": value} if value else None), shortened


def derive_record(
    source: dict[str, Any], assistant_turn: int, turns: list[tuple[str, str]],
    system_text: str | None, args: argparse.Namespace,
) -> dict[str, Any] | None:
    current_user, current_assistant = turns[-1]
    messages: list[dict[str, str]] = []
    compacted = False
    if len(turns) == 1:
        keep_system = system_text is not None and include_system(source["id"])
        if keep_system:
            system, shortened = compact_message("system", system_text, args.max_system_chars)
            compacted |= shortened
            if system is not None:
                messages.append(system)
        user, shortened = compact_message("user", current_user, args.max_user_chars)
        compacted |= shortened
        assistant, shortened = compact_message(
            "assistant", current_assistant, args.max_assistant_chars
        )
        compacted |= shortened
        if user is None or assistant is None:
            return None
        messages.extend((user, assistant))
        derivation = "first-complete-turn"
        system_included = keep_system
    else:
        previous_user, previous_assistant = turns[-2]
        for role, content in (
            ("user", previous_user),
            ("assistant", previous_assistant),
            ("user", current_user),
            ("assistant", current_assistant),
        ):
            maximum = (
                args.context_user_chars if role == "user" else args.context_assistant_chars
            )
            message, shortened = compact_message(role, content, maximum)
            compacted |= shortened
            if message is None:
                return None
            messages.append(message)
        derivation = "last-two-complete-turns"
        system_included = False

    if not messages:
        return None
    result: dict[str, Any] = {
        "id": f"{source['id']}:assistant-turn-{assistant_turn}",
        "source": source["source"],
        "license": source["license"],
        "approval_status": source["approval_status"],
        "messages": messages,
        "provenance": {
            "derived_from_id": source["id"],
            "assistant_turn": assistant_turn,
            "derivation": derivation,
            "target_context": 512,
            "original_system_included": system_included,
            "content_compacted": compacted,
            "limits": {
                "user_characters": args.max_user_chars,
                "assistant_characters": args.max_assistant_chars,
                "system_characters": args.max_system_chars,
                "context_user_characters": args.context_user_chars,
                "context_assistant_characters": args.context_assistant_chars,
            },
        },
    }
    for key in ("category", "topic", "sub_topic"):
        if isinstance(source.get(key), str) and source[key]:
            result[key] = source[key]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-user-chars", type=int, default=160)
    parser.add_argument("--max-assistant-chars", type=int, default=300)
    parser.add_argument("--max-system-chars", type=int, default=60)
    parser.add_argument("--context-user-chars", type=int, default=80)
    parser.add_argument("--context-assistant-chars", type=int, default=140)
    args = parser.parse_args()
    if min(
        args.max_user_chars, args.max_assistant_chars, args.max_system_chars,
        args.context_user_chars, args.context_assistant_chars,
    ) < 32:
        raise SystemExit("all character limits must be at least 32")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    total = 0
    derived = 0
    dropped = 0
    seen_ids: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source_file, args.output.open("x", encoding="utf-8") as output_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            source = json.loads(line)
            total += 1
            source_id = source.get("id")
            if not isinstance(source_id, str) or not source_id:
                raise SystemExit(f"missing source ID on line {line_number}")
            messages = source.get("messages")
            if not isinstance(messages, list):
                raise SystemExit(f"missing messages array on line {line_number}")
            system_text: str | None = None
            turns: list[tuple[str, str]] = []
            previous_user: str | None = None
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                if role == "system" and system_text is None:
                    system_text = content
                elif role == "user":
                    previous_user = content
                elif role == "assistant" and previous_user is not None:
                    turns.append((previous_user, content))
                    record = derive_record(
                        source, len(turns), turns, system_text, args
                    )
                    if record is None:
                        dropped += 1
                        continue
                    if record["id"] in seen_ids:
                        raise SystemExit(f"duplicate derived ID: {record['id']}")
                    seen_ids.add(record["id"])
                    output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    derived += 1
                    previous_user = None
    print(json.dumps({"source_conversations": total, "derived_examples": derived, "dropped": dropped}))


if __name__ == "__main__":
    main()
