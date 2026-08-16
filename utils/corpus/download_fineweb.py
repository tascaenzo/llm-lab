#!/usr/bin/env python3
"""Scarica shard italiane di FineWeb-2 con provenienza verificabile.

FineWeb-2 pubblica l'italiano in `data/ita_Latn/train`, 85 shard Parquet da
circa 4 GiB. L'intero sottoinsieme supera i 330 GiB, ma una sola shard contiene
gia' molti piu' token di quanti ne servano al corpus: lo script scarica il
numero di shard richiesto, non tutto.

Usa soltanto la libreria standard. La lettura del Parquet richiede pyarrow ed e'
compito di normalize_source.py: qui si scaricano byte e si registra da dove
vengono.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATASET = "HuggingFaceFW/fineweb-2"
CONFIGURATION = "ita_Latn"
LICENSE = "ODC-By-1.0"
TREE_URL = f"https://huggingface.co/api/datasets/{DATASET}/tree/main/data/{CONFIGURATION}/train"
RESOLVE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
USER_AGENT = "llm-lab-corpus/1"
CHUNK = 1 << 20


def request(url: str, headers: dict[str, str] | None = None):
    merged = {"User-Agent": USER_AGENT}
    merged.update(headers or {})
    return urlopen(Request(url, headers=merged), timeout=120)


def list_shards() -> list[dict]:
    """Elenca le shard disponibili con dimensione dichiarata dall'API."""
    entries = json.loads(request(TREE_URL).read())
    shards = [entry for entry in entries if entry["path"].endswith(".parquet")]
    shards.sort(key=lambda entry: entry["path"])
    return shards


def format_bytes(value: int) -> str:
    return f"{value / (1 << 30):.2f} GiB" if value >= (1 << 30) else f"{value / (1 << 20):.0f} MiB"


def download_shard(path: str, expected_size: int, destination: Path) -> str:
    """Scarica una shard riprendendo un download parziale e ne ritorna lo sha256."""
    digest = hashlib.sha256()
    already = destination.stat().st_size if destination.exists() else 0
    if already == expected_size and expected_size > 0:
        with destination.open("rb") as existing:
            for block in iter(lambda: existing.read(CHUNK), b""):
                digest.update(block)
        print(f"  {destination.name}: gia' completa", file=sys.stderr)
        return digest.hexdigest()
    if already > expected_size:
        destination.unlink()
        already = 0

    headers = {"Range": f"bytes={already}-"} if already else {}
    if already:
        with destination.open("rb") as existing:
            for block in iter(lambda: existing.read(CHUNK), b""):
                digest.update(block)
        print(f"  {destination.name}: riprendo da {format_bytes(already)}", file=sys.stderr)

    response = request(f"{RESOLVE_URL}/{path}", headers)
    received = already
    started = time.monotonic()
    last_report = 0.0
    with destination.open("ab" if already else "wb") as output:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            output.write(block)
            digest.update(block)
            received += len(block)
            now = time.monotonic()
            if now - last_report >= 1.0:
                elapsed = max(now - started, 1e-9)
                speed = (received - already) / elapsed / (1 << 20)
                percent = 100.0 * received / expected_size if expected_size else 0.0
                print(
                    f"\r  {destination.name}: {percent:5.1f}% "
                    f"{format_bytes(received)}/{format_bytes(expected_size)} {speed:6.1f} MiB/s",
                    end="",
                    file=sys.stderr,
                )
                last_report = now
    print("", file=sys.stderr)
    if expected_size and received != expected_size:
        raise RuntimeError(f"{path}: attesi {expected_size} byte, ricevuti {received}")
    return digest.hexdigest()


def write_provenance(directory: Path, records: list[dict]) -> None:
    """Registra origine, licenza, checksum e data: e' cio' che rende ripetibile il corpus."""
    document = {
        "schema": "llm-lab-source-v1",
        "source": "fineweb2-ita",
        "dataset": DATASET,
        "configuration": CONFIGURATION,
        "license": LICENSE,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shards": records,
    }
    (directory / "source.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", type=int, default=1, help="numero di shard da scaricare (default: 1)"
    )
    parser.add_argument(
        "--first", type=int, default=0, help="indice della prima shard (default: 0)"
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/fineweb2-ita"),
        help="directory di destinazione",
    )
    parser.add_argument("--list", action="store_true", help="elenca le shard e termina")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        shards = list_shards()
    except (HTTPError, URLError) as error:
        print(f"Elenco delle shard non riuscito: {error}", file=sys.stderr)
        return 1

    total = sum(shard.get("size", 0) for shard in shards)
    print(
        f"{DATASET} / {CONFIGURATION}: {len(shards)} shard, {format_bytes(total)} in totale",
        file=sys.stderr,
    )
    if args.list:
        for index, shard in enumerate(shards):
            print(f"{index:3d}  {shard['path'].split('/')[-1]:24s} {format_bytes(shard.get('size', 0))}")
        return 0

    if args.first < 0 or args.first >= len(shards) or args.shards < 1:
        print("Intervallo di shard non valido.", file=sys.stderr)
        return 1
    selected = shards[args.first : args.first + args.shards]
    args.destination.mkdir(parents=True, exist_ok=True)
    print(
        f"Scarico {len(selected)} shard "
        f"({format_bytes(sum(shard.get('size', 0) for shard in selected))}) in {args.destination}",
        file=sys.stderr,
    )

    records = []
    for shard in selected:
        name = shard["path"].split("/")[-1]
        size = shard.get("size", 0)
        try:
            checksum = download_shard(shard["path"], size, args.destination / name)
        except (HTTPError, URLError, RuntimeError, OSError) as error:
            print(f"Download di {name} non riuscito: {error}", file=sys.stderr)
            return 1
        records.append({"file": name, "path": shard["path"], "bytes": size, "sha256": checksum})
        write_provenance(args.destination, records)

    print(f"Completato: {len(records)} shard in {args.destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
