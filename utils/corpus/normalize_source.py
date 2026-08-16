#!/usr/bin/env python3
"""Converte una fonte scaricata nello schema `documents.jsonl` del corpus.

Ogni fonte arriva in un formato diverso; il corpus ne accetta uno solo:

    {"id": "fonte:chiave", "source": "...", "license": "...", "url": "...", "text": "..."}

L'`id` deve essere prefissato dalla fonte. Senza namespace due documenti di
fonti diverse possono collidere, e poiche' lo split del dataset e' derivato
dall'ID la collisione produce assegnazioni non deterministiche.

Wikipedia continua a passare da extract_wikipedia.py, che e' il normalizzatore
di quella fonte. Qui si trattano le fonti Parquet, cioe' FineWeb-2.

Dipendenza: pyarrow, richiesto solo per leggere il Parquet. E' uno strumento di
preparazione dati offline: il runtime di training resta senza dipendenze.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

SOURCES = {
    "fineweb2-ita": {
        "prefix": "fineweb2",
        "license": "ODC-By-1.0",
        "language": "ita",
    }
}


def import_parquet():
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        print(
            "pyarrow non installato. E' richiesto solo dalla preparazione del corpus:\n"
            "  python3 -m venv .venv && .venv/bin/pip install pyarrow",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return parquet


def clean_text(text: str) -> str:
    """Normalizza soltanto i caratteri di controllo e gli spazi di fine riga.

    Accenti, apostrofi e punteggiatura restano intatti: sono informazione utile
    per un tokenizer byte-level, e riscriverli sarebbe una decisione arbitraria
    sull'italiano invece che una pulizia.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        character
        for character in text
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
    )
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def document_identifier(prefix: str, raw: str) -> str:
    """FineWeb-2 usa `<urn:uuid:...>`; conserviamo solo l'UUID."""
    key = raw.strip()
    if key.startswith("<") and key.endswith(">"):
        key = key[1:-1]
    if key.startswith("urn:uuid:"):
        key = key[len("urn:uuid:") :]
    return f"{prefix}:{key}"


def convert_parquet(
    paths: list[Path],
    descriptor: dict,
    output: Path,
    minimum_characters: int,
    minimum_language_score: float,
) -> dict:
    parquet = import_parquet()
    statistics = {
        "rows_read": 0,
        "documents_written": 0,
        "skipped_short": 0,
        "skipped_language": 0,
        "skipped_empty": 0,
        "characters_written": 0,
    }
    seen: set[str] = set()
    with output.open("w", encoding="utf-8") as sink:
        for path in paths:
            reader = parquet.ParquetFile(path)
            columns = ["id", "text", "url", "language", "language_score"]
            available = set(reader.schema_arrow.names)
            columns = [column for column in columns if column in available]
            for batch in reader.iter_batches(batch_size=4096, columns=columns):
                records = batch.to_pylist()
                for record in records:
                    statistics["rows_read"] += 1
                    score = record.get("language_score")
                    if record.get("language") != descriptor["language"] or (
                        score is not None and score < minimum_language_score
                    ):
                        statistics["skipped_language"] += 1
                        continue
                    text = clean_text(record.get("text") or "")
                    if not text:
                        statistics["skipped_empty"] += 1
                        continue
                    if len(text) < minimum_characters:
                        statistics["skipped_short"] += 1
                        continue
                    identifier = document_identifier(descriptor["prefix"], record.get("id") or "")
                    if identifier in seen:
                        continue
                    seen.add(identifier)
                    sink.write(
                        json.dumps(
                            {
                                "id": identifier,
                                "source": descriptor["source"],
                                "license": descriptor["license"],
                                "url": record.get("url") or "",
                                "text": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    statistics["documents_written"] += 1
                    statistics["characters_written"] += len(text)
            print(f"  {path.name}: {statistics['documents_written']} documenti", file=sys.stderr)
    return statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=sorted(SOURCES), help="fonte da normalizzare")
    parser.add_argument("--input", type=Path, required=True, help="directory con i file scaricati")
    parser.add_argument("--output", type=Path, required=True, help="documents.jsonl da scrivere")
    parser.add_argument(
        "--min-characters", type=int, default=200, help="lunghezza minima del testo pulito"
    )
    parser.add_argument(
        "--min-language-score",
        type=float,
        default=0.9,
        help="soglia minima di confidenza sulla lingua",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    descriptor = dict(SOURCES[args.source])
    descriptor["source"] = args.source
    paths = sorted(args.input.glob("*.parquet"))
    if not paths:
        print(f"Nessun file .parquet in {args.input}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Normalizzo {len(paths)} file da {args.source}", file=sys.stderr)
    statistics = convert_parquet(
        paths, descriptor, args.output, args.min_characters, args.min_language_score
    )

    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "schema": "llm-lab-normalized-source-v1",
                "source": args.source,
                "license": descriptor["license"],
                "inputs": [path.name for path in paths],
                "filters": {
                    "min_characters": args.min_characters,
                    "min_language_score": args.min_language_score,
                    "language": descriptor["language"],
                },
                "statistics": statistics,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(statistics, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
