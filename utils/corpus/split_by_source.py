#!/usr/bin/env python3
"""Separa un documents.jsonl deduplicato in un file per fonte.

La deduplicazione deve vedere tutte le fonti insieme, perche' il suo scopo e'
proprio trovare le sovrapposizioni fra l'una e l'altra, e produce quindi un
flusso unico. Il mixer lavora invece per fonte, dovendo applicare a ciascuna la
propria quota. Questo passaggio fa da raccordo.

Usa soltanto la libreria standard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="documents.jsonl deduplicato")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory di destinazione")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"File non trovato: {args.input}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    handles: dict[str, object] = {}
    counts: dict[str, int] = {}
    try:
        with args.input.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                line = line.strip()
                if not line:
                    continue
                document = json.loads(line)
                name = document.get("source")
                if not isinstance(name, str) or not name:
                    print(f"Documento senza fonte alla riga {line_number}", file=sys.stderr)
                    return 1
                if name not in handles:
                    handles[name] = (args.output_dir / f"{name}.jsonl").open(
                        "w", encoding="utf-8"
                    )
                    counts[name] = 0
                handles[name].write(line + "\n")
                counts[name] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} documenti -> {args.output_dir / f'{name}.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
