#!/usr/bin/env python3
"""Estrae un corpus testuale riproducibile da un dump pages-articles di Wikipedia."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import html
import json
import multiprocessing
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as element_tree
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Tuple


MEBIBYTE = 1024 * 1024
WRITE_BUFFER_BYTES = MEBIBYTE
CLEANING_CHUNK_SIZE = 16
COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
REF_PATTERN = re.compile(r"<ref\b[^>]*>.*?</ref\s*>", re.IGNORECASE | re.DOTALL)
SELF_CLOSING_REF_PATTERN = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)
TABLE_PATTERN = re.compile(r"\{\|.*?\|\}", re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")
EXTERNAL_LINK_PATTERN = re.compile(r"\[(?:https?://|//)[^\s\]]+(?:\s+([^\]]+))?\]")
HEADING_PATTERN = re.compile(r"^={2,}\s*(.*?)\s*={2,}$", re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES_PATTERN = re.compile(r"\n{3,}")
WIKILINK_PATTERN = re.compile(r"\[\[([^\[\]\n]+)\]\]")
REDIRECT_PATTERN = re.compile(r"^\s*#(?:redirect|rinvia)\b", re.IGNORECASE)
INVALID_NAME_PATTERN = re.compile(r"[^a-z0-9-]")

RawPage = Tuple[str, str, str]
CleanPage = Tuple[str, str, str]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def direct_child(element: element_tree.Element, name: str) -> Optional[element_tree.Element]:
    for child in element:
        if local_name(child.tag) == name:
            return child
    return None


def direct_child_text(element: element_tree.Element, name: str) -> str:
    child = direct_child(element, name)
    return "" if child is None or child.text is None else child.text


def strip_balanced(text: str, opening: str, closing: str) -> str:
    result = []
    cursor = 0
    while True:
        start = text.find(opening, cursor)
        if start < 0:
            result.append(text[cursor:])
            return "".join(result)

        result.append(text[cursor:start])
        depth = 1
        cursor = start + len(opening)
        while cursor < len(text) and depth > 0:
            next_opening = text.find(opening, cursor)
            next_closing = text.find(closing, cursor)
            if next_closing < 0:
                return "".join(result)
            if 0 <= next_opening < next_closing:
                depth += 1
                cursor = next_opening + len(opening)
            else:
                depth -= 1
                cursor = next_closing + len(closing)


def replace_wikilink(match: re.Match) -> str:
    fields = match.group(1).split("|")
    target = fields[0].strip().lower()
    if target.startswith(("categoria:", "category:", "file:", "immagine:", "image:")):
        return " "
    return fields[-1].strip()


def clean_wikitext(text: str) -> str:
    text = COMMENT_PATTERN.sub(" ", text)
    text = REF_PATTERN.sub(" ", text)
    text = SELF_CLOSING_REF_PATTERN.sub(" ", text)
    text = strip_balanced(text, "{{", "}}")
    text = TABLE_PATTERN.sub(" ", text)
    text = WIKILINK_PATTERN.sub(replace_wikilink, text)
    text = EXTERNAL_LINK_PATTERN.sub(lambda match: match.group(1) or " ", text)
    text = HEADING_PATTERN.sub(r"\1", text)
    text = text.replace("'''", "").replace("''", "")
    text = TAG_PATTERN.sub(" ", text)
    text = html.unescape(text).replace("\u00a0", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return BLANK_LINES_PATTERN.sub("\n\n", text).strip()


def clean_page(page: RawPage) -> CleanPage:
    page_id, title, raw_text = page
    return page_id, title, clean_wikitext(raw_text)


def safe_name(value: str) -> str:
    result = INVALID_NAME_PATTERN.sub("-", value.lower()).strip("-")
    if not result:
        raise ValueError("--name deve contenere almeno una lettera o una cifra")
    return result


def discover_dump(project_root: Path) -> Path:
    candidates = sorted((project_root / "data" / "raw" / "wikipedia-it").glob("*-pages-articles.xml.bz2"))
    if len(candidates) != 1:
        raise RuntimeError("specificare --input: non e' stato trovato un solo dump pages-articles")
    return candidates[0]


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit_index = 0
    while value >= 1024.0 and unit_index + 1 < len(units):
        value /= 1024.0
        unit_index += 1
    return f"{value:.1f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


class CountingReader:
    def __init__(self, input_file) -> None:
        self.input_file = input_file
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self.input_file.read(size)
        self.bytes_read += len(data)
        return data

    def __getattr__(self, name: str):
        return getattr(self.input_file, name)


class ProgressReporter:
    def __init__(self, total_bytes: int) -> None:
        self.total_bytes = total_bytes
        self.bytes_read = 0
        self.started_at = time.monotonic()
        self.last_update_at = 0.0
        self.enabled = sys.stderr.isatty()

    def update(
        self,
        bytes_read: Optional[int],
        pages_seen: int,
        documents_written: int,
        force: bool = False,
    ) -> None:
        if bytes_read is not None:
            self.bytes_read = bytes_read
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_update_at < 0.5:
            return

        elapsed = now - self.started_at
        fraction = min(1.0, self.bytes_read / self.total_bytes) if self.total_bytes else 0.0
        width = 26
        completed = int(fraction * width)
        bar = "#" * completed + "-" * (width - completed)
        speed = self.bytes_read / elapsed if elapsed else 0.0
        remaining = (self.total_bytes - self.bytes_read) / speed if speed else 0.0
        eta = format_duration(remaining) if speed else "--:--"
        print(
            f"\rEstrazione [{bar}] {fraction * 100:5.1f}% "
            f"{format_bytes(self.bytes_read)}/{format_bytes(self.total_bytes)} "
            f"{format_bytes(speed)}/s ETA {eta} | "
            f"pagine {pages_seen:,}, documenti {documents_written:,}",
            end="",
            file=sys.stderr,
            flush=True,
        )
        self.last_update_at = now

    def finish(self) -> None:
        if self.enabled:
            print(file=sys.stderr)


def iter_raw_pages(input_path: Path, statistics: dict, reporter: ProgressReporter) -> Iterator[RawPage]:
    with input_path.open("rb") as input_file:
        counting_reader = CountingReader(input_file)
        with bz2.BZ2File(counting_reader, "rb") as dump_file:
            reporter.update(0, 0, 0, force=True)
            parser = element_tree.iterparse(dump_file, events=("start", "end"))
            root = None
            for event, page in parser:
                if event == "start" and root is None:
                    root = page
                    continue
                if event != "end" or local_name(page.tag) != "page":
                    continue

                statistics["pages_seen"] += 1
                reporter.update(
                    counting_reader.bytes_read,
                    statistics["pages_seen"],
                    statistics["documents_written"],
                )
                namespace = direct_child_text(page, "ns")
                page_id = direct_child_text(page, "id")
                revision = direct_child(page, "revision")
                raw_text = "" if revision is None else direct_child_text(revision, "text")
                is_redirect = direct_child(page, "redirect") is not None or bool(
                    REDIRECT_PATTERN.match(raw_text)
                )
                title = direct_child_text(page, "title")
                page.clear()
                if root is not None:
                    root.clear()

                if namespace != "0":
                    continue
                statistics["pages_main_namespace"] += 1
                if is_redirect:
                    statistics["redirects_skipped"] += 1
                    continue
                if not page_id:
                    continue
                yield page_id, title, raw_text
            reporter.update(
                counting_reader.bytes_read,
                statistics["pages_seen"],
                statistics["documents_written"],
                force=True,
            )


def iter_clean_pages(pages: Iterator[RawPage], workers: int) -> Iterator[CleanPage]:
    if workers == 1:
        for page in pages:
            yield clean_page(page)
        return

    with multiprocessing.Pool(processes=workers) as pool:
        yield from pool.imap(clean_page, pages, chunksize=CLEANING_CHUNK_SIZE)


class PartWriter:
    def __init__(self, output_dir: Path, part_size_bytes: int) -> None:
        self.output_dir = output_dir
        self.part_size_bytes = part_size_bytes
        self.current_file = None
        self.current_temporary_path = None
        self.current_final_path = None
        self.current_size = 0
        self.part_index = 0
        self.paths = []

    def open_part(self) -> None:
        self.current_final_path = self.output_dir / f"part-{self.part_index:03d}.txt"
        self.current_temporary_path = self.current_final_path.with_suffix(".txt.part")
        self.current_file = self.current_temporary_path.open(
            "w", encoding="utf-8", buffering=WRITE_BUFFER_BYTES
        )
        self.current_size = 0

    def close_part(self) -> None:
        if self.current_file is None:
            return
        self.current_file.close()
        self.current_temporary_path.replace(self.current_final_path)
        self.paths.append(self.current_final_path)
        self.part_index += 1
        self.current_file = None
        self.current_temporary_path = None
        self.current_final_path = None
        self.current_size = 0

    def write_document(self, text: str) -> None:
        encoded_size = len(text.encode("utf-8")) + 2
        if self.current_file is None:
            self.open_part()
        elif self.current_size > 0 and self.current_size + encoded_size > self.part_size_bytes:
            self.close_part()
            self.open_part()
        self.current_file.write(text)
        self.current_file.write("\n\n")
        self.current_size += encoded_size

    def close(self) -> None:
        self.close_part()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_workers = max(1, (os.cpu_count() or 2) - 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="dump .xml.bz2; se omesso viene rilevato in data/raw")
    parser.add_argument("--name", default="italiano-wikipedia-v1", help="nome stabile del corpus")
    parser.add_argument("--clean-root", type=Path, default=project_root / "data" / "clean")
    parser.add_argument("--derived-root", type=Path, default=project_root / "data" / "derived")
    parser.add_argument("--min-characters", type=int, default=200)
    parser.add_argument("--part-size-mib", type=int, default=256)
    parser.add_argument("--workers", type=int, default=default_workers)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.min_characters < 1:
        raise ValueError("--min-characters deve essere positivo")
    if args.part_size_mib < 1:
        raise ValueError("--part-size-mib deve essere positivo")
    if args.workers < 1:
        raise ValueError("--workers deve essere positivo")


def output_is_empty(path: Path) -> bool:
    return not path.exists() or not any(path.iterdir())


def load_source_provenance(input_path: Path) -> object:
    source_path = input_path.parent / "source.json"
    if not source_path.is_file():
        return None
    return json.loads(source_path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    deduplication = None
    reporter = None

    try:
        validate_args(args)
        name = safe_name(args.name)
        input_path = args.input if args.input is not None else discover_dump(project_root)
        input_path = input_path.resolve()
        if not input_path.is_file():
            raise RuntimeError(f"dump non trovato: {input_path}")
        reporter = ProgressReporter(input_path.stat().st_size)

        clean_dir = args.clean_root / name
        derived_dir = args.derived_root / name / "tokenizer-input"
        if not output_is_empty(clean_dir) or not output_is_empty(derived_dir):
            raise RuntimeError("la destinazione contiene gia' dati; scegli --name diverso")
        clean_dir.mkdir(parents=True, exist_ok=True)
        derived_dir.mkdir(parents=True, exist_ok=True)

        documents_path = clean_dir / "documents.jsonl"
        temporary_documents_path = documents_path.with_suffix(".jsonl.part")
        deduplication_path = clean_dir / "deduplication.sqlite.part"
        part_writer = PartWriter(derived_dir, args.part_size_mib * MEBIBYTE)
        statistics = {
            "pages_seen": 0,
            "pages_main_namespace": 0,
            "redirects_skipped": 0,
            "too_short_skipped": 0,
            "duplicates_skipped": 0,
            "documents_written": 0,
            "text_bytes_written": 0,
        }

        deduplication = sqlite3.connect(deduplication_path)
        deduplication.execute("PRAGMA journal_mode=OFF")
        deduplication.execute("PRAGMA synchronous=OFF")
        deduplication.execute("CREATE TABLE document_hashes (digest BLOB PRIMARY KEY)")

        with temporary_documents_path.open(
            "w", encoding="utf-8", buffering=WRITE_BUFFER_BYTES
        ) as documents_file:
            for page_id, title, text in iter_clean_pages(
                iter_raw_pages(input_path, statistics, reporter), args.workers
            ):
                if len(text) < args.min_characters:
                    statistics["too_short_skipped"] += 1
                    continue

                digest = hashlib.sha256(text.encode("utf-8")).digest()
                cursor = deduplication.execute("INSERT OR IGNORE INTO document_hashes VALUES (?)", (digest,))
                if cursor.rowcount == 0:
                    statistics["duplicates_skipped"] += 1
                    continue

                record = {
                    "id": f"wikipedia-it:{page_id}",
                    "source": "wikipedia-it",
                    "license": "CC BY-SA",
                    "url": f"https://it.wikipedia.org/?curid={page_id}",
                    "title": title,
                    "text": text,
                }
                documents_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                documents_file.write("\n")
                part_writer.write_document(text)
                statistics["documents_written"] += 1
                statistics["text_bytes_written"] += len(text.encode("utf-8"))

                if statistics["documents_written"] % 10000 == 0:
                    deduplication.commit()
                    reporter.update(None, statistics["pages_seen"], statistics["documents_written"])

        deduplication.commit()
        deduplication.close()
        deduplication = None
        deduplication_path.unlink()
        part_writer.close()
        temporary_documents_path.replace(documents_path)
        reporter.update(
            input_path.stat().st_size,
            statistics["pages_seen"],
            statistics["documents_written"],
            force=True,
        )
        reporter.finish()
        reporter = None

        manifest = {
            "schema": "llm-lab-corpus-v1",
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "name": "Wikipedia in italiano",
                "input": relative_to_project(input_path, project_root),
                "provenance": load_source_provenance(input_path),
            },
            "selection": {"namespace": 0, "skip_redirects": True, "min_characters": args.min_characters, "deduplicate": True},
            "cleaning": {"version": "wikitext-basic-v1", "description": "rimuove markup e conserva testo visibile"},
            "execution": {"workers": args.workers, "cleaning_chunk_size": CLEANING_CHUNK_SIZE},
            "outputs": {
                "documents": relative_to_project(documents_path, project_root),
                "tokenizer_input": [relative_to_project(path, project_root) for path in part_writer.paths],
            },
            "statistics": statistics,
        }
        (clean_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Corpus creato: {clean_dir}")
        print(f"Documenti: {statistics['documents_written']:,}")
        print(f"Testo pulito: {statistics['text_bytes_written']:,} byte")
        print(f"Parti per il tokenizer: {len(part_writer.paths)}")
        return 0
    except (OSError, ValueError, sqlite3.Error, element_tree.ParseError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1
    finally:
        if deduplication is not None:
            deduplication.close()
        if reporter is not None:
            reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
