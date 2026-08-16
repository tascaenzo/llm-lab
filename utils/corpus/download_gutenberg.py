#!/usr/bin/env python3
"""Scarica i libri italiani di Project Gutenberg in testo semplice.

E' una fonte opzionale di prosa lunga: romanzi, saggi e poesia da usare solo
dopo una verifica dei diritti per l'Italia, documentata in un'allowlist.

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

try:
    from progress import ProgressBar
except ModuleNotFoundError:
    from utils.corpus.progress import ProgressBar

CATALOG_URL = "https://gutendex.com/books?languages=it"
MIRRORS = [
    "https://gutenberg.pglaf.org/cache/epub/{identifier}/pg{identifier}.txt",
    "http://aleph.gutenberg.org/cache/epub/{identifier}/pg{identifier}.txt",
]
USER_AGENT = "llm-lab-corpus/1"
LICENSE = "Public domain (curated allowlist; verify jurisdiction)"


def fetch(url: str, timeout: int = 60) -> bytes:
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout).read()


def load_allowlist(path: Path) -> dict[int, dict]:
    """Loads reviewed Italian rights records keyed by Project Gutenberg ID.

    Project Gutenberg's catalogue is useful discovery metadata, not a rights
    clearance for an Italian training corpus.  Keeping the evidence alongside
    each selected title makes the provenance auditable and prevents a broad,
    incorrect "all Gutenberg is public domain" claim.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"allowlist non leggibile: {path} ({error})") from error
    if (
        payload.get("schema") != "llm-lab-gutenberg-allowlist-v1"
        or payload.get("jurisdiction") != "IT"
    ):
        raise ValueError("l'allowlist deve dichiarare schema v1 e jurisdiction 'IT'")
    entries = payload.get("books")
    if not isinstance(entries, list) or not entries:
        raise ValueError("l'allowlist non contiene alcun libro verificato")
    result: dict[int, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), int):
            raise ValueError("ogni record dell'allowlist deve avere un id intero")
        if entry.get("rights") != "public-domain-it" or not isinstance(entry.get("evidence"), str):
            raise ValueError("ogni record deve dichiarare public-domain-it e una evidenza")
        if entry["id"] in result:
            raise ValueError(f"id Gutenberg duplicato nell'allowlist: {entry['id']}")
        result[entry["id"]] = entry
    return result


def load_catalog(limit: int, allowlist: dict[int, dict]) -> list[dict]:
    """Percorre le pagine di gutendex fino al numero di libri richiesto."""
    books: list[dict] = []
    url = CATALOG_URL
    progress = ProgressBar("Catalogo Gutenberg", unit="libri")
    while url and len(books) < limit:
        payload = json.loads(fetch(url))
        for book in payload["results"]:
            identifier = book.get("id")
            # Gutendex exposes the Project Gutenberg copyright marker as well;
            # require both that metadata and the repository's IT review.
            if identifier not in allowlist or book.get("copyright") is not False:
                continue
            books.append(
                {
                    "id": identifier,
                    "title": book.get("title", ""),
                    "authors": [author.get("name", "") for author in book.get("authors", [])],
                    "rights": allowlist[identifier]["rights"],
                    "rights_evidence": allowlist[identifier]["evidence"],
                    "rights_verified_at": allowlist[identifier].get("verified_at", ""),
                }
            )
            if len(books) >= limit:
                break
        url = payload.get("next")
        progress.update(len(books), "titoli verificati trovati")
    progress.finish(len(books), "ricerca completata")
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
    parser.add_argument(
        "--allowlist",
        type=Path,
        required=True,
        help="JSON di titoli verificati per il pubblico dominio in Italia",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    texts = args.destination / "texts"
    texts.mkdir(exist_ok=True)

    try:
        allowlist = load_allowlist(args.allowlist)
        books = load_catalog(args.limit, allowlist)
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as error:
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
    progress = ProgressBar("Download Gutenberg", len(books), "libri")
    for index, book in enumerate(books, 1):
        target = texts / f"{book['id']}.txt"
        if target.is_file() and book["id"] in known:
            skipped += 1
            progress.update(index, f"scaricati {downloaded}, gia' presenti {skipped}, falliti {failed}")
            continue
        result = download_book(book["id"])
        if result is None:
            failed += 1
            progress.update(index, f"scaricati {downloaded}, gia' presenti {skipped}, falliti {failed}")
            continue
        payload, url = result
        target.write_bytes(payload)
        book["url"] = url
        book["bytes"] = len(payload)
        known[book["id"]] = book
        downloaded += 1
        progress.update(index, f"scaricati {downloaded}, gia' presenti {skipped}, falliti {failed}")
        # Il mirror e' un servizio gratuito offerto da volontari: si aspetta.
        time.sleep(args.delay)
    progress.finish(len(books), f"scaricati {downloaded}, gia' presenti {skipped}, falliti {failed}")

    catalog_path.write_text(
        json.dumps(
            {
                "schema": "llm-lab-source-v1",
                "source": "gutenberg-ita",
                "license": LICENSE,
                "catalog": CATALOG_URL,
                "allowlist": str(args.allowlist),
                "jurisdiction": "IT",
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
