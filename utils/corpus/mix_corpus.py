#!/usr/bin/env python3
"""Compone il corpus finale applicando le quote dichiarate per fonte.

Le quote sono espresse in token, ma il tokenizer del corpus nuovo non esiste
ancora: si addestra dopo, e proprio su questo corpus. Si usa quindi il tokenizer
gia' disponibile come **metro di misura**, non come tokenizer finale: si campiona
ogni fonte, si misurano i byte per token con `llm-lab tokenizer evaluate` e si
converte la quota in un budget di byte.

L'errore che questo introduce e' stato misurato: il tokenizer addestrato su
Wikipedia rende 2,764 byte per token sul proprio corpus e 2,659 su FineWeb-2,
cioe' il 3,8%. Calibrando per fonte quell'errore si annulla quasi del tutto e
resta ben dentro la tolleranza del 2% sulle quote.

La selezione dei documenti e' deterministica: si ordina per hash dell'ID, quindi
lo stesso input produce sempre lo stesso corpus, indipendentemente dall'ordine
dei file di partenza.

Usa soltanto la libreria standard.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
UINT64_MASK = 0xFFFFFFFFFFFFFFFF
SAMPLE_BYTES = 4 << 20
DEFAULT_BYTES_PER_TOKEN = 2.7


def fnv1a_64(text: str) -> int:
    value = FNV_OFFSET
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * FNV_PRIME) & UINT64_MASK
    return value


def measure_bytes_per_token(binary: Path, tokenizer: Path, documents: Path) -> float:
    """Misura i byte per token della fonte con il tokenizer esistente."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as sample:
        written = 0
        with documents.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = json.loads(line).get("text") or ""
                sample.write(text + "\n")
                written += len(text.encode("utf-8"))
                if written >= SAMPLE_BYTES:
                    break
        sample_path = Path(sample.name)
    try:
        result = subprocess.run(
            [str(binary), "tokenizer", "evaluate", str(tokenizer), str(SAMPLE_BYTES),
             str(sample_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "tokenizer evaluate non riuscito")
        return float(json.loads(result.stdout)["bytes_per_token"])
    finally:
        sample_path.unlink(missing_ok=True)


def index_documents(path: Path) -> tuple[list[tuple[int, int, int]], int]:
    """Ritorna (hash, offset, byte del testo) per documento, piu' il totale."""
    entries = []
    total = 0
    with path.open("rb") as handle:
        offset = handle.tell()
        for line in handle:
            if line.strip():
                record = json.loads(line)
                size = len((record.get("text") or "").encode("utf-8"))
                entries.append((fnv1a_64(record.get("id") or ""), offset, size))
                total += size
            offset = handle.tell()
    entries.sort()
    return entries, total


def parse_pairs(values: list[str], name: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{name} non valido: {value}. Usa NOME=VALORE.")
        key, _, item = value.partition("=")
        result[key] = item
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--quota", action="append", required=True, metavar="NAME=FRACTION")
    parser.add_argument("--output", type=Path, required=True, help="documents.jsonl finale")
    parser.add_argument("--manifest", type=Path, required=True, help="manifest.json del corpus")
    parser.add_argument("--name", default="italiano-v3", help="nome del corpus")
    parser.add_argument("--target-tokens", type=int, default=3_000_000_000)
    parser.add_argument("--tokenizer", type=Path, help="tokenizer usato solo per la calibrazione")
    parser.add_argument("--llm-lab", type=Path, help="binario llm-lab per la calibrazione")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = parse_pairs(args.input, "--input")
    quotas = {key: float(value) for key, value in parse_pairs(args.quota, "--quota").items()}
    if set(inputs) != set(quotas):
        print("Le fonti di --input e --quota non coincidono.", file=sys.stderr)
        return 1
    total_quota = sum(quotas.values())
    if abs(total_quota - 1.0) > 1e-6:
        print(f"Le quote sommano a {total_quota:.4f}, non a 1.", file=sys.stderr)
        return 1

    plan = []
    for name, path in inputs.items():
        source = Path(path)
        if not source.is_file():
            print(f"File non trovato: {source}", file=sys.stderr)
            return 1
        if args.tokenizer and args.llm_lab:
            try:
                ratio = measure_bytes_per_token(args.llm_lab, args.tokenizer, source)
            except (RuntimeError, KeyError, ValueError) as error:
                print(f"Calibrazione di {name} non riuscita: {error}", file=sys.stderr)
                return 1
        else:
            ratio = DEFAULT_BYTES_PER_TOKEN
        entries, available_bytes = index_documents(source)
        available_tokens = available_bytes / ratio
        wanted_tokens = args.target_tokens * quotas[name]
        # Una fonte piu' povera della sua quota si prende per intero: meglio un
        # corpus piu' piccolo e dichiarato che un mix falsato in silenzio.
        taken_tokens = min(wanted_tokens, available_tokens)
        plan.append(
            {
                "source": name,
                "path": source,
                "entries": entries,
                "bytes_per_token": round(ratio, 4),
                "available_tokens": int(available_tokens),
                "wanted_tokens": int(wanted_tokens),
                "byte_budget": int(taken_tokens * ratio),
            }
        )
        short = " (sotto quota)" if taken_tokens < wanted_tokens - 1 else ""
        print(
            f"  {name}: {ratio:.3f} byte/token, disponibili {available_tokens/1e6:.1f}M token, "
            f"richiesti {wanted_tokens/1e6:.1f}M{short}",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    statistics = []
    with args.output.open("w", encoding="utf-8") as sink:
        for item in plan:
            written_bytes = 0
            written_documents = 0
            with item["path"].open("rb") as handle:
                for _, offset, size in item["entries"]:
                    if written_bytes >= item["byte_budget"]:
                        break
                    handle.seek(offset)
                    sink.write(handle.readline().decode("utf-8"))
                    written_bytes += size
                    written_documents += 1
            statistics.append(
                {
                    "source": item["source"],
                    "documents": written_documents,
                    "bytes": written_bytes,
                    "estimated_tokens": int(written_bytes / item["bytes_per_token"]),
                    "bytes_per_token": item["bytes_per_token"],
                    "available_tokens": item["available_tokens"],
                }
            )
            print(
                f"  {item['source']}: {written_documents} documenti, "
                f"{statistics[-1]['estimated_tokens']/1e6:.1f}M token stimati",
                file=sys.stderr,
            )

    total_tokens = sum(entry["estimated_tokens"] for entry in statistics)
    for entry in statistics:
        entry["achieved_quota"] = (
            round(entry["estimated_tokens"] / total_tokens, 4) if total_tokens else 0.0
        )
        entry["requested_quota"] = quotas[entry["source"]]

    manifest = {
        "schema": "llm-lab-corpus-v1",
        "name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outputs": {"documents": str(args.output)},
        "target_tokens": args.target_tokens,
        "estimated_tokens": total_tokens,
        "documents": sum(entry["documents"] for entry in statistics),
        "sources": statistics,
        "notes": (
            "I token sono stimati con il tokenizer indicato, calibrato per fonte. "
            "Il conteggio esatto arriva da dataset prepare dopo l'addestramento "
            "del tokenizer definitivo."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"\nCorpus {args.name}: {manifest['documents']} documenti, "
        f"{total_tokens/1e9:.3f} miliardi di token stimati",
        file=sys.stderr,
    )
    for entry in statistics:
        print(
            f"  {entry['source']:16} quota richiesta {entry['requested_quota']:.0%}, "
            f"ottenuta {entry['achieved_quota']:.1%}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
