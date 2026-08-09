#!/usr/bin/env python3
"""Download e verifica del dump pages-articles di Wikipedia in italiano."""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://dumps.wikimedia.org/itwiki/latest"
DEFAULT_FILENAME = "itwiki-latest-pages-articles.xml.bz2"
CHUNK_SIZE = 1024 * 1024


def read_url(url: str):
    request = Request(url, headers={"User-Agent": "llm-lab-corpus/1"})
    return urlopen(request, timeout=60)


def expected_md5(checksums_url: str, filename: str) -> tuple[str, str]:
    """Return the checksum and the dated filename listed by Wikimedia."""
    suffix_parts = filename.split("-pages-", 1)
    expected_suffix = f"-pages-{suffix_parts[1]}" if len(suffix_parts) == 2 else filename

    with read_url(checksums_url) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            fields = line.split()
            listed_filename = fields[1].lstrip("*") if len(fields) == 2 else ""
            if listed_filename == filename or listed_filename.endswith(expected_suffix):
                digest = fields[0].lower()
                if len(digest) == 32 and all(char in "0123456789abcdef" for char in digest):
                    return digest, listed_filename
    raise RuntimeError(f"checksum MD5 non trovato per {filename}")


def download(url: str, destination: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    temporary = destination.with_suffix(destination.suffix + ".part")

    with read_url(url) as response, temporary.open("wb") as output:
        total = response.headers.get("Content-Length")
        if total is not None:
            print(f"Download: {int(total):,} byte")
        else:
            print("Download: dimensione non dichiarata")

        downloaded = 0
        while chunk := response.read(CHUNK_SIZE):
            output.write(chunk)
            md5.update(chunk)
            sha256.update(chunk)
            downloaded += len(chunk)
            print(f"\rScaricati: {downloaded:,} byte", end="", flush=True)

    print()
    return md5.hexdigest(), sha256.hexdigest()


def write_provenance(destination: Path, url: str, md5: str, sha256: str) -> None:
    provenance = {
        "schema": "llm-lab-corpus-source-v1",
        "source": "Wikipedia in italiano",
        "license": "CC BY-SA (consultare i termini Wikimedia)",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "download_url": url,
        "file": destination.name,
        "source_checksum": {"algorithm": "md5", "value": md5},
        "sha256": sha256,
    }
    (destination.parent / "source.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=f"{DEFAULT_BASE_URL}/{DEFAULT_FILENAME}",
        help="URL del dump da scaricare",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "raw" / "wikipedia-it",
        help="directory di destinazione (default: data/raw/wikipedia-it)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="mostra il checksum pubblicato senza scaricare il dump",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_url = args.url
    requested_filename = requested_url.rsplit("/", 1)[-1]
    dump_version = requested_filename.split("-pages-", 1)[0]
    dump_directory = requested_url.rsplit("/", 1)[0]
    checksums_url = dump_directory + f"/{dump_version}-md5sums.txt"

    try:
        expected, source_filename = expected_md5(checksums_url, requested_filename)
        print(f"MD5 atteso: {expected}")
        print(f"File della fonte: {source_filename}")
        if args.check:
            return 0

        source_version = source_filename.split("-pages-", 1)[0]
        if dump_directory.endswith("/latest"):
            project_directory = dump_directory.rsplit("/", 1)[0]
            snapshot_date = source_version.rsplit("-", 1)[-1]
            dump_url = f"{project_directory}/{snapshot_date}/{source_filename}"
        else:
            dump_url = f"{dump_directory}/{source_filename}"

        destination = args.output_dir / source_filename
        if destination.exists():
            print(f"File gia' presente: {destination}", file=sys.stderr)
            print("Rimuovilo o scegli --output-dir diverso.", file=sys.stderr)
            return 1

        args.output_dir.mkdir(parents=True, exist_ok=True)
        actual_md5, actual_sha256 = download(dump_url, destination)
    except URLError as error:
        if isinstance(error.reason, ssl.SSLCertVerificationError):
            print("Errore: Python non trova i certificati CA attendibili.", file=sys.stderr)
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            print(
                "Su macOS con Python.org esegui: "
                f'open "/Applications/Python {python_version}/Install Certificates.command"',
                file=sys.stderr,
            )
        else:
            print(f"Errore: {error}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    if actual_md5 != expected:
        print(f"MD5 ottenuto: {actual_md5}", file=sys.stderr)
        print("Checksum non valido: il file temporaneo e' stato conservato.", file=sys.stderr)
        return 1

    destination.with_suffix(destination.suffix + ".part").replace(destination)
    write_provenance(destination, dump_url, actual_md5, actual_sha256)
    print(f"Verifica completata: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
