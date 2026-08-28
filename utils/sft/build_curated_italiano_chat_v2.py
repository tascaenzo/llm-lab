#!/usr/bin/env python3
"""Build a conservative, traceable chat-SFT corpus from v1.

The source corpus is left untouched.  This builder deliberately excludes the
templated v2 batch and reduces the imported v3 conversations to their first
complete exchange.  It then applies transparent, deterministic quality guards
appropriate for a small general-purpose chat model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TEMPLATED_SOURCE = "local-synthetic:chatgpt-generated-v2"
IMPORTED_SOURCE = "local-synthetic:ita_conversations_v3"
GENERIC_SYSTEM = "Sei un assistente utile, chiaro e pratico."
WHITESPACE = re.compile(r"\s+")
URL = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
QUESTION = re.compile(r"[?？]")
PROFANITY = re.compile(r"\b(?:cazzo|cazzi|stronz\w*|vaffanculo|maroni|porca\s+puttana)\b", re.IGNORECASE)

# These domains need dedicated, reviewed examples instead of generic synthetic
# answers.  A token match is intentionally conservative: the corpus is for a
# first chat-behaviour fine-tune, not a domain expert model.
SENSITIVE_DOMAIN = re.compile(
    r"\b(?:"
    r"medic[oa]|medicina|farmac[oi]|terapi[ae]|diagnos[it]|sintom[io]|"
    r"malatti[ae]|riabilitaz\w*|fisioterap\w*|dolor\w*|infortuni\w*|"
    r"psicolog\w*|psichiatr\w*|suicid\w*|depressione|"
    r"avvocat\w*|legal[ei]|leggi|legge|decreto|d\.lgs|reat[oi]|tribunal\w*|"
    r"contratt\w*|multa|fisc\w*|tass\w*|"
    r"elezion\w*|partit\w*|politic\w*|governo|parlamento|unione europea|"
    r"migratori\w*|"
    r"investiment\w*|azioni|criptovalut\w*|bitcoin|mutuo|prestito|borsa|finanza\w*"
    r")\b",
    re.IGNORECASE,
)
UNSAFE_INSTRUCTION = re.compile(
    r"\b(?:costruire una bomba|fabbricare una bomba|fare una bomba|"
    r"aggirare la sicurezza|eludere i controlli|rubare|truffare|"
    r"farsi del male|uccidersi)\b",
    re.IGNORECASE,
)


def normalise(value: str) -> str:
    return WHITESPACE.sub(" ", value).strip()


def first_pair(messages: Any) -> tuple[str, str] | None:
    """Return the first user/assistant pair without inventing a dialogue."""
    if not isinstance(messages, list):
        return None
    user: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content = normalise(content)
        if not content:
            continue
        if role == "user":
            user = content
        elif role == "assistant" and user is not None:
            return user, content
    return None


def rejection_reason(user: str, assistant: str) -> str | None:
    if not 24 <= len(user) <= 650:
        return "user_length"
    if not 80 <= len(assistant) <= 1_250:
        return "assistant_length"
    if not QUESTION.search(user):
        return "not_a_question"
    combined = f"{user}\n{assistant}"
    if URL.search(combined):
        return "url"
    if PROFANITY.search(combined):
        return "profanity"
    if SENSITIVE_DOMAIN.search(combined):
        return "sensitive_domain"
    if UNSAFE_INSTRUCTION.search(combined):
        return "unsafe_instruction"
    # A malformed response is especially harmful for a 75M chat model.
    if assistant.count("?") > 5 or len(re.findall(r"\b(\w+)(?:\s+\1){3,}\b", assistant, re.IGNORECASE)):
        return "degenerate_response"
    return None


def build_record(source: dict[str, Any], user: str, assistant: str) -> dict[str, Any]:
    record_id = source["id"]
    # derive_context512_chat.py retains a stable 20% sample of systems.  Put
    # the canonical system here for every record so that that sampling happens
    # exactly once and has the intended frequency in the training file.
    messages: list[dict[str, str]] = [
        {"role": "system", "content": GENERIC_SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    result: dict[str, Any] = {
        "id": f"{record_id}:curated-first-turn",
        "source": source["source"],
        "license": source["license"],
        "approval_status": source["approval_status"],
        "messages": messages,
        "provenance": {
            "derived_from_id": record_id,
            "derivation": "curated-first-complete-turn",
            "builder": "utils/sft/build_curated_italiano_chat_v2.py",
            "source_message_count": len(source["messages"]),
            "generic_system_available": True,
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
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite an existing output or report")

    statistics: Counter[str] = Counter()
    accepted_sources: Counter[str] = Counter()
    seen_pairs: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source_file, args.output.open("x", encoding="utf-8") as output_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            source = json.loads(line)
            statistics["source_conversations"] += 1
            if source.get("source") == TEMPLATED_SOURCE:
                statistics["excluded_templated_batch"] += 1
                continue
            if not all(isinstance(source.get(key), str) and source[key] for key in ("id", "source", "license", "approval_status")):
                raise SystemExit(f"invalid required metadata on line {line_number}")
            pair = first_pair(source.get("messages"))
            if pair is None:
                statistics["rejected_no_complete_pair"] += 1
                continue
            user, assistant = pair
            reason = rejection_reason(user, assistant)
            if reason is not None:
                statistics[f"rejected_{reason}"] += 1
                continue
            pair_fingerprint = hashlib.sha256(f"{user}\0{assistant}".encode("utf-8")).hexdigest()
            if pair_fingerprint in seen_pairs:
                statistics["rejected_duplicate_pair"] += 1
                continue
            seen_pairs.add(pair_fingerprint)
            record = build_record(source, user, assistant)
            output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            statistics["accepted"] += 1
            accepted_sources[source["source"]] += 1

    report = {
        "schema": "italiano-chat-corpus-v2-curation-report-v1",
        "input": str(args.input),
        "output": str(args.output),
        "selection": {
            "excluded_source": TEMPLATED_SOURCE,
            "all_selected_records": "first complete user/assistant pair only",
            "imported_v3_policy": "first pair only; later turns are excluded",
            "system_prompt": "stable 20% generic sample",
            "sensitive_domains": "excluded pending a dedicated reviewed corpus",
        },
        "counts": dict(sorted(statistics.items())),
        "accepted_by_source": dict(sorted(accepted_sources.items())),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
