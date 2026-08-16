#!/usr/bin/env python3
"""Deriva input tokenizer train-only da un corpus documents.jsonl esistente."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from progress import ProgressBar
except ModuleNotFoundError:
    from utils.corpus.progress import ProgressBar


MEBIBYTE = 1024 * 1024
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1


def fnv1a_64(text: str) -> int:
    value = FNV_OFFSET
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * FNV_PRIME) & UINT64_MASK
    return value


def document_split(document_id: str) -> str:
    bucket = fnv1a_64(document_id) % 10000
    if bucket < 9000:
        return "train"
    if bucket < 9500:
        return "validation"
    return "test"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(MEBIBYTE):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


class PartWriter:
    def __init__(self, output_dir: Path, maximum_bytes: int):
        self.output_dir = output_dir
        self.maximum_bytes = maximum_bytes
        self.index = 0
        self.size = 0
        self.file = None
        self.final_path = None
        self.temporary_path = None
        self.paths: list[Path] = []

    def _open(self) -> None:
        self.final_path = self.output_dir / f"part-{self.index:03d}.txt"
        self.temporary_path = self.output_dir / f"part-{self.index:03d}.txt.part"
        self.file = self.temporary_path.open("w", encoding="utf-8", buffering=MEBIBYTE)
        self.size = 0

    def _close(self) -> None:
        if self.file is None:
            return
        self.file.close()
        self.temporary_path.replace(self.final_path)
        self.paths.append(self.final_path)
        self.index += 1
        self.file = None
        self.final_path = None
        self.temporary_path = None
        self.size = 0

    def write(self, text: str) -> None:
        encoded_size = len(text.encode("utf-8")) + 2
        if self.file is None:
            self._open()
        elif self.size > 0 and self.size + encoded_size > self.maximum_bytes:
            self._close()
            self._open()
        self.file.write(text)
        self.file.write("\n\n")
        self.size += encoded_size

    def close(self) -> None:
        self._close()


def prepare(corpus_manifest_path: Path, output_dir: Path, output_manifest_path: Path,
            part_size_bytes: int, project_root: Path) -> dict:
    if part_size_bytes < 1:
        raise ValueError("la dimensione delle parti deve essere positiva")
    if output_manifest_path.exists():
        raise RuntimeError(f"il manifest di output esiste gia': {output_manifest_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"la directory di output non e' vuota: {output_dir}")

    source_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    documents_value = source_manifest.get("outputs", {}).get("documents")
    corpus_name = source_manifest.get("name")
    if not isinstance(documents_value, str) or not isinstance(corpus_name, str):
        raise RuntimeError("manifest del corpus incompleto")
    documents_path = Path(documents_value)
    if not documents_path.is_absolute():
        documents_path = project_root / documents_path
    if not documents_path.is_file():
        raise RuntimeError(f"documents.jsonl non trovato: {documents_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PartWriter(output_dir, part_size_bytes)
    split_counts = {"train": 0, "validation": 0, "test": 0}
    train_bytes = 0
    progress = ProgressBar("Preparazione corpus tokenizer", documents_path.stat().st_size, "bytes")
    bytes_read = 0
    try:
        with documents_path.open("r", encoding="utf-8") as documents:
            for line_number, line in enumerate(documents, start=1):
                bytes_read += len(line.encode("utf-8"))
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"JSON non valido alla riga {line_number}") from error
                document_id = record.get("id")
                text = record.get("text")
                if not isinstance(document_id, str) or not document_id or not isinstance(text, str):
                    raise RuntimeError(f"documento incompleto alla riga {line_number}")
                split = document_split(document_id)
                split_counts[split] += 1
                if split == "train":
                    writer.write(text)
                    train_bytes += len(text.encode("utf-8"))
                progress.update(bytes_read, f"documenti {line_number:,}; train {split_counts['train']:,}")
    finally:
        writer.close()
    progress.finish(bytes_read, f"documenti train {split_counts['train']:,}")
    if not writer.paths:
        raise RuntimeError("nessun documento train disponibile")

    manifest = {
        "schema": "llm-lab-tokenizer-corpus-v1",
        "name": f"{corpus_name}-train-only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": portable_path(corpus_manifest_path, project_root),
        "source_manifest_sha256": sha256_file(corpus_manifest_path),
        "split": {
            "algorithm": "fnv1a-64-mod-10000",
            "selected": "train",
            "train_buckets": [0, 8999],
            "validation_buckets": [9000, 9499],
            "test_buckets": [9500, 9999],
        },
        "outputs": {
            "documents": portable_path(documents_path, project_root),
            "tokenizer_input": [portable_path(path, project_root) for path in writer.paths],
            "tokenizer_input_split": "train",
        },
        "statistics": {
            "documents": split_counts,
            "train_text_bytes": train_bytes,
            "parts": len(writer.paths),
        },
    }
    temporary_manifest = output_manifest_path.with_suffix(output_manifest_path.suffix + ".part")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(output_manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-manifest", type=Path,
        default=project_root / "data" / "clean" / "italiano-wikipedia-v1" / "manifest.json"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project_root / "data" / "derived" / "italiano-wikipedia-v1" / "tokenizer-train-input"
    )
    parser.add_argument(
        "--output-manifest", type=Path,
        default=project_root / "data" / "derived" / "italiano-wikipedia-v1" /
        "tokenizer-train-manifest.json"
    )
    parser.add_argument("--part-size-mib", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    try:
        manifest = prepare(args.corpus_manifest, args.output_dir, args.output_manifest,
                           args.part_size_mib * MEBIBYTE, project_root)
        print(f"Manifest tokenizer train-only: {args.output_manifest}")
        print(f"Documenti train: {manifest['statistics']['documents']['train']:,}")
        print(f"Parti: {manifest['statistics']['parts']}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
