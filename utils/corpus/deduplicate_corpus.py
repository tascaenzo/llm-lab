#!/usr/bin/env python3
"""Deduplica documenti fra piu' fonti, prima dell'assegnazione degli split.

Il rischio non e' lo spreco di spazio: e' la contaminazione. FineWeb-2 contiene
Wikipedia, perche' e' mirrorata ovunque nel web. Se lo stesso testo finisce sia
in train sia in validation, la loss di validation risulta ottimistica e la curva
non lo mostra.

Il duplicato tipico non e' identico: e' un articolo Wikipedia annegato in una
pagina con menu, footer e pubblicita'. L'hash del documento intero non lo vede.
Si confrontano quindi le **frasi**, i cui confini sono definiti dal contenuto e
non si spostano quando cambia il boilerplate attorno.

Due livelli:

1. SHA-256 del testo normalizzato: cattura le copie integrali a costo nullo.
2. Sovrapposizione di frasi: un documento e' un quasi-duplicato quando una
   frazione sufficiente delle sue frasi e' gia' comparsa in una fonte di
   priorita' maggiore, o prima nella stessa fonte.

Le fonti si elaborano in ordine di priorita' decrescente, quindi in una
collisione sopravvive sempre la copia migliore.

L'indice delle frasi e' un filtro di Bloom: un set Python di decine di milioni
di hash costerebbe gigabyte, il filtro costa circa 1,2 byte per frase. I falsi
positivi valgono la soglia dichiarata e non spostano il risultato, perche' la
decisione dipende da una frazione di frasi e non da una singola.

Usa soltanto la libreria standard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

try:
    from progress import ProgressBar
except ModuleNotFoundError:
    from utils.corpus.progress import ProgressBar

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?:;])\s+|\n+")
WHITESPACE = re.compile(r"\s+")


class BloomFilter:
    """Filtro di Bloom con parametri derivati da capacita' e tasso di errore."""

    def __init__(self, capacity: int, false_positive_rate: float) -> None:
        capacity = max(capacity, 1)
        bits = max(int(-capacity * math.log(false_positive_rate) / (math.log(2) ** 2)), 8)
        self.bit_count = bits
        self.hash_count = max(1, min(16, round(bits / capacity * math.log(2))))
        self.bits = bytearray((bits + 7) // 8)
        self.inserted = 0

    def _positions(self, digest: bytes):
        # Un solo digest da 32 byte alimenta tutte le funzioni hash: e' la
        # tecnica di Kirsch-Mitzenmacher, indistinguibile in pratica da k hash
        # indipendenti e molto piu' economica.
        first = int.from_bytes(digest[:8], "little")
        second = int.from_bytes(digest[8:16], "little") | 1
        for index in range(self.hash_count):
            yield (first + index * second) % self.bit_count

    def add(self, digest: bytes) -> None:
        for position in self._positions(digest):
            self.bits[position >> 3] |= 1 << (position & 7)
        self.inserted += 1

    def __contains__(self, digest: bytes) -> bool:
        return all(
            self.bits[position >> 3] & (1 << (position & 7)) for position in self._positions(digest)
        )

    def memory_bytes(self) -> int:
        return len(self.bits)


def normalize(text: str) -> str:
    """Forma di confronto: minuscole, spazi collassati, accenti conservati."""
    text = unicodedata.normalize("NFC", text).lower()
    return WHITESPACE.sub(" ", text).strip()


def sentences(text: str, minimum_characters: int) -> list[str]:
    """Frasi abbastanza lunghe da essere informative.

    Le righe corte sono voci di menu, date, didascalie: comparirebbero identiche
    in migliaia di pagine non imparentate e produrrebbero falsi duplicati.
    """
    result = []
    for candidate in SENTENCE_BOUNDARY.split(text):
        normalized = normalize(candidate)
        if len(normalized) >= minimum_characters:
            result.append(normalized)
    return result


def digest(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def count_lines(path: Path) -> int:
    total = 0
    with path.open("rb") as handle:
        for _ in handle:
            total += 1
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="fonte in ordine di priorita' decrescente, es. wikipedia=data/clean/wiki.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True, help="documents.jsonl deduplicato")
    parser.add_argument("--report", type=Path, help="report JSON della deduplicazione")
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help="frazione di frasi gia' viste oltre la quale il documento e' un duplicato",
    )
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=3,
        help="sotto questa lunghezza si applica solo la deduplicazione esatta",
    )
    parser.add_argument(
        "--min-sentence-characters",
        type=int,
        default=40,
        help="lunghezza minima di una frase perche' entri nel confronto",
    )
    parser.add_argument(
        "--false-positive-rate", type=float, default=0.01, help="tasso di errore del filtro"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = []
    for entry in args.input:
        if "=" not in entry:
            print(f"Fonte non valida: {entry}. Usa NOME=PERCORSO.", file=sys.stderr)
            return 1
        name, _, path = entry.partition("=")
        source_path = Path(path)
        if not source_path.is_file():
            print(f"File non trovato: {source_path}", file=sys.stderr)
            return 1
        sources.append((name, source_path))

    # Un filtro per fonte gia' elaborata: serve ad attribuire lo scarto alla
    # fonte che possiede l'originale, non solo a contarlo.
    filters: dict[str, BloomFilter] = {}
    exact_seen: dict[bytes, str] = {}
    statistics = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    total_documents = sum(count_lines(path) for _, path in sources)
    progress = ProgressBar("Deduplicazione", total_documents, "documenti")
    processed_documents = 0

    with args.output.open("w", encoding="utf-8") as sink:
        for name, path in sources:
            estimated_sentences = max(path.stat().st_size // 120, 1024)
            filters[name] = BloomFilter(estimated_sentences, args.false_positive_rate)
            entry = {
                "source": name,
                "documents_read": 0,
                "documents_kept": 0,
                "characters_kept": 0,
                "dropped_exact": 0,
                "dropped_overlap": 0,
                "dropped_by_source": {},
                "filter_megabytes": round(filters[name].memory_bytes() / (1 << 20), 1),
            }
            print(
                f"{name}: filtro da {entry['filter_megabytes']} MiB "
                f"per ~{estimated_sentences} frasi",
                file=sys.stderr,
            )
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    document = json.loads(line)
                    text = document.get("text") or ""
                    entry["documents_read"] += 1
                    processed_documents += 1
                    progress.update(
                        processed_documents,
                        f"tenuti {total_kept:,}; fonte {name}",
                    )

                    exact = digest(normalize(text))
                    owner = exact_seen.get(exact)
                    if owner is not None:
                        entry["dropped_exact"] += 1
                        entry["dropped_by_source"][owner] = (
                            entry["dropped_by_source"].get(owner, 0) + 1
                        )
                        continue

                    parts = sentences(text, args.min_sentence_characters)
                    digests = [digest(part) for part in parts]
                    duplicate_owner = None
                    if len(digests) >= args.min_sentences:
                        # Si attribuisce alla prima fonte che supera la soglia,
                        # nell'ordine di priorita' con cui sono state indicizzate.
                        for other_name, other_filter in filters.items():
                            seen = sum(1 for item in digests if item in other_filter)
                            if seen / len(digests) >= args.overlap_threshold:
                                duplicate_owner = other_name
                                break
                    if duplicate_owner is not None:
                        entry["dropped_overlap"] += 1
                        entry["dropped_by_source"][duplicate_owner] = (
                            entry["dropped_by_source"].get(duplicate_owner, 0) + 1
                        )
                        continue

                    exact_seen[exact] = name
                    for item in digests:
                        filters[name].add(item)
                    sink.write(json.dumps(document, ensure_ascii=False) + "\n")
                    entry["documents_kept"] += 1
                    entry["characters_kept"] += len(text)
                    total_kept += 1
                    if entry["documents_read"] % 100000 == 0:
                        print(
                            f"  {name}: {entry['documents_read']} letti, "
                            f"{entry['documents_kept']} tenuti",
                            file=sys.stderr,
                        )
            statistics.append(entry)
            print(
                f"  {name}: {entry['documents_kept']}/{entry['documents_read']} tenuti, "
                f"{entry['dropped_exact']} esatti + {entry['dropped_overlap']} sovrapposti scartati",
                file=sys.stderr,
            )

    progress.finish(processed_documents, f"tenuti {total_kept:,}")

    report = {
        "schema": "llm-lab-deduplication-v1",
        "parameters": {
            "overlap_threshold": args.overlap_threshold,
            "min_sentences": args.min_sentences,
            "min_sentence_characters": args.min_sentence_characters,
            "false_positive_rate": args.false_positive_rate,
        },
        "sources": statistics,
        "documents_kept": total_kept,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(report["sources"], indent=2, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
