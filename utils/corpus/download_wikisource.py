#!/usr/bin/env python3
"""Scarica il dump di Wikisource italiano.

Wikisource raccoglie testi di pubblico dominio gia' trascritti e revisionati:
letteratura, saggistica e documenti storici. E' la seconda fonte narrativa del
corpus e la piu' economica da acquisire, perche' usa la stessa infrastruttura di
dump di Wikipedia.

Riusa `download_wikipedia.py` cambiando soltanto il progetto: la verifica dei
checksum, la ripresa e il file di provenienza sono gli stessi.

Usa soltanto la libreria standard.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

PROJECT = "itwikisource"
BASE_URL = f"https://dumps.wikimedia.org/{PROJECT}/latest"
FILENAME = f"{PROJECT}-latest-pages-articles.xml.bz2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=f"{BASE_URL}/{FILENAME}", help="URL del dump")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/wikisource-it"),
        help="directory di destinazione",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    downloader = Path(__file__).with_name("download_wikipedia.py")
    sys.argv = [
        str(downloader),
        "--url",
        arguments.url,
        "--output-dir",
        str(arguments.destination),
    ]
    try:
        runpy.run_path(str(downloader), run_name="__main__")
    except SystemExit as exit_status:
        return int(exit_status.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
