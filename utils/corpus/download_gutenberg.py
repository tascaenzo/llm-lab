#!/usr/bin/env python3
"""Scarica i libri italiani di Project Gutenberg in testo semplice.

E' la fonte che porta la prosa lunga: romanzi, saggi e poesia di pubblico
dominio, cioe' l'unico registro narrativo che ne' Wikipedia ne' il web
forniscono.

Il catalogo arriva da gutendex.com, un indice pubblico di Gutenberg che risponde
in JSON. I testi arrivano dai mirror ufficiali: `www.gutenberg.org` limita il
download automatico e ne raccomanda l'uso.

Ogni libro e' un file `<id>.txt` piu' una riga in `catalog.json` con titolo,
autore e URL, cosi' la provenienza resta verificabile documento per documento.

Usa soltanto la libreria standard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

CATALOG_URL = "https://gutendex.com/books?languages=it"
MIRRORS = [
    "https://gutenberg.pglaf.org/cache/epub/{identifier}/pg{identifier}.txt",
    "http://aleph.gutenberg.org/cache/epub/{identifier}/pg{identifier}.txt",
]
USER_AGENT = "llm-lab-corpus/1"
LICENSE = "Public domain (Project Gutenberg)"


def fetch(url: str, timeout: int = 60) -> bytes:
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout).read()


def load_catalog(limit: int) -> list[dict]:
    """Percorre le pagine di gutendex fino al numero di libri richiesto."""
    books: list[dict] = []
    url = CATALOG_URL
    while url and len(books) < limit:
        payload = json.loads(fetch(url))
        for book in payload["results"]:
            books.append(
                {
                    "id": book["id"],
                    "title": book.get("title", ""),
                    "authors": [author.get("name", "") for author in book.get("authors", [])],
                }
            )
            if len(books) >= limit:
                break
        url = payload.get("next")
        print(f"  catalogo: {len(books)} libri", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return books


def download_book(identifier: int) -> tuple[bytes, str] | None:
    for template in MIRRORS:
        url = template.format(identifier=identifier)
        try:
            return fetch(url, timeout=45), url
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000, help="numero massimo di libri")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/gutenberg-ita"),
        help="directory di destinazione",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="pausa fra un download e l'altro, per non gravare sul mirror",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    texts = args.destination / "texts"
    texts.mkdir(exist_ok=True)

    try:
        books = load_catalog(args.limit)
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        print(f"Catalogo non raggiungibile: {error}", file=sys.stderr)
        return 1
    print(f"Catalogo italiano: {len(books)} libri", file=sys.stderr)

    catalog_path = args.destination / "catalog.json"
    known = {}
    if catalog_path.is_file():
        known = {
            entry["id"]: entry
            for entry in json.loads(catalog_path.read_text(encoding="utf-8"))["books"]
        }

    downloaded, skipped, failed = 0, 0, 0
    for index, book in enumerate(books, 1):
        target = texts / f"{book['id']}.txt"
        if target.is_file() and book["id"] in known:
            skipped += 1
            continue
        result = download_book(book["id"])
        if result is None:
            failed += 1
            continue
        payload, url = result
        target.write_bytes(payload)
        book["url"] = url
        book["bytes"] = len(payload)
        known[book["id"]] = book
        downloaded += 1
        print(
            f"  {index}/{len(books)} scaricati {downloaded}, saltati {skipped}, falliti {failed}",
            end="\r",
            file=sys.stderr,
        )
        # Il mirror e' un servizio gratuito offerto da volontari: si aspetta.
        time.sleep(args.delay)
    print(file=sys.stderr)

    catalog_path.write_text(
        json.dumps(
            {
                "schema": "llm-lab-source-v1",
                "source": "gutenberg-ita",
                "license": LICENSE,
                "catalog": CATALOG_URL,
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "books": sorted(known.values(), key=lambda entry: entry["id"]),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Completato: {downloaded} scaricati, {skipped} gia' presenti, {failed} non disponibili",
        file=sys.stderr,
    )
    return 0 if known else 1


if __name__ == "__main__":
    raise SystemExit(main())
