#!/usr/bin/env python3
"""Converte i libri di Project Gutenberg nello schema `documents.jsonl`.

Ogni file scaricato contiene, oltre al testo, una licenza di alcune migliaia di
parole in inglese, ripetuta identica in tutti i libri. Lasciarla dentro
significherebbe insegnare al modello quel boilerplate mille volte e sporcare la
statistica della lingua: i marcatori `*** START` e `*** END` la delimitano ed e'
quello che si taglia.

Un libro e' inoltre troppo lungo per una finestra di contesto: viene spezzato in
sezioni di alcune migliaia di caratteri, tagliando su righe vuote, cosi' ogni
documento resta un blocco di prosa continua. Le sezioni conservano l'ID del
libro nel proprio, per poter risalire alla fonte.

Usa soltanto la libreria standard.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

try:
    from progress import ProgressBar
except ModuleNotFoundError:
    from utils.corpus.progress import ProgressBar

START_MARKER = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
END_MARKER = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
# I file piu' vecchi chiudono senza asterischi: "End of Project Gutenberg's <titolo>".
LEGACY_END_MARKER = re.compile(r"^.{0,40}End of (?:the )?Project Gutenberg.*$", re.I | re.M)
PRODUCED_BY = re.compile(r"^\s*(?:produced by|e-?text prepared by|transcriber).*$", re.I | re.M)
BLANK_LINES = re.compile(r"\n{3,}")


def strip_license(text: str) -> str | None:
    """Tiene solo cio' che sta fra i marcatori di inizio e fine."""
    start = START_MARKER.search(text)
    if start is not None:
        text = text[start.end() :]
    # Un file puo' contenere entrambe le forme di chiusura, la vecchia prima
    # della nuova: vale la piu' a sinistra, altrimenti resta in mezzo il
    # colophon del trascrittore.
    ends = [match for match in (END_MARKER.search(text), LEGACY_END_MARKER.search(text)) if match]
    end = min(ends, key=lambda match: match.start()) if ends else None
    if end is not None:
        text = text[: end.start()]
    if start is None and end is None:
        # Senza marcatori non si distingue il testo dalla licenza: meglio
        # scartare il libro che inquinare il corpus.
        return None
    return text


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = PRODUCED_BY.sub("", text)
    text = "".join(
        character
        for character in text
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
    )
    lines = [line.rstrip() for line in text.split("\n")]
    return BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def sections(text: str, target_characters: int) -> list[str]:
    """Spezza su righe vuote, accumulando fino alla dimensione richiesta."""
    result = []
    current: list[str] = []
    size = 0
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        current.append(paragraph)
        size += len(paragraph) + 2
        if size >= target_characters:
            result.append("\n\n".join(current))
            current = []
            size = 0
    if current:
        result.append("\n\n".join(current))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/gutenberg-ita"), help="directory scaricata"
    )
    parser.add_argument("--output", type=Path, required=True, help="documents.jsonl da scrivere")
    parser.add_argument(
        "--section-characters", type=int, default=8000, help="dimensione indicativa di una sezione"
    )
    parser.add_argument("--min-characters", type=int, default=500, help="sezione minima accettata")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog_path = args.input / "catalog.json"
    if not catalog_path.is_file():
        print(f"Catalogo non trovato: {catalog_path}", file=sys.stderr)
        return 1
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    books = {entry["id"]: entry for entry in catalog["books"]}
    license_name = catalog.get("license", "Public domain (Project Gutenberg)")

    statistics = {
        "books_read": 0,
        "books_without_markers": 0,
        "documents_written": 0,
        "characters_written": 0,
    }
    paths = sorted((args.input / "texts").glob("*.txt"))
    progress = ProgressBar("Normalizzazione Gutenberg", len(paths), "libri")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for book_index, path in enumerate(paths, 1):
            identifier = int(path.stem)
            book = books.get(identifier, {})
            statistics["books_read"] += 1
            body = strip_license(path.read_text(encoding="utf-8", errors="replace"))
            if body is None:
                statistics["books_without_markers"] += 1
                progress.update(book_index, f"documenti {statistics['documents_written']:,}")
                continue
            body = clean(body)
            for index, section in enumerate(sections(body, args.section_characters)):
                if len(section) < args.min_characters:
                    continue
                sink.write(
                    json.dumps(
                        {
                            "id": f"gutenberg:{identifier}-{index:04d}",
                            "source": "gutenberg-ita",
                            "license": book.get("rights", license_name),
                            "url": book.get("url", ""),
                            "title": book.get("title", ""),
                            "rights_evidence": book.get("rights_evidence", ""),
                            "text": section,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                statistics["documents_written"] += 1
                statistics["characters_written"] += len(section)
            progress.update(book_index, f"documenti {statistics['documents_written']:,}")

    progress.finish(len(paths), f"documenti {statistics['documents_written']:,}")

    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema": "llm-lab-normalized-source-v1",
                "source": "gutenberg-ita",
                "license": license_name,
                "filters": {
                    "section_characters": args.section_characters,
                    "min_characters": args.min_characters,
                    "strip_project_gutenberg_license": True,
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
