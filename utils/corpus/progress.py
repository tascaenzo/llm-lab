"""Progress reporting shared by offline corpus tools, with no dependencies."""

from __future__ import annotations

import sys
import time
import os


def format_quantity(value: int, unit: str) -> str:
    if unit == "bytes":
        amount = float(value)
        for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
            if amount < 1024 or suffix == "TiB":
                return f"{amount:.1f} {suffix}" if suffix != "B" else f"{int(amount)} B"
            amount /= 1024
    return f"{value:,} {unit}"


class ProgressBar:
    """A TTY bar, or periodic log records when output is redirected."""

    def __init__(self, label: str, total: int | None = None, unit: str = "items"):
        self.label = label
        self.total = total if total and total > 0 else None
        self.unit = unit
        self.started_at = time.monotonic()
        self.last_report_at = 0.0
        self.interactive = sys.stderr.isatty() and os.environ.get("LLM_LAB_PROGRESS_LOG") != "1"
        self.finished = False

    def update(self, value: int, detail: str = "", force: bool = False) -> None:
        now = time.monotonic()
        interval = 0.5 if self.interactive else 5.0
        if not force and now - self.last_report_at < interval:
            return
        elapsed = max(now - self.started_at, 1e-9)
        current = format_quantity(value, self.unit)
        speed = format_quantity(int(value / elapsed), self.unit)
        suffix = f" | {detail}" if detail else ""
        if self.total is not None:
            fraction = min(1.0, value / self.total)
            bar_size = 24
            completed = int(bar_size * fraction)
            message = (
                f"{self.label} [{'#' * completed}{'-' * (bar_size - completed)}] "
                f"{fraction * 100:5.1f}% {current}/{format_quantity(self.total, self.unit)} "
                f"{speed}/s{suffix}"
            )
        else:
            message = f"{self.label}: {current} ({speed}/s){suffix}"
        ending = "\r" if self.interactive and not force else "\n"
        print(f"\r{message}" if self.interactive and not force else message, end=ending,
              file=sys.stderr, flush=True)
        self.last_report_at = now

    def finish(self, value: int, detail: str = "") -> None:
        if self.finished:
            return
        self.update(value, detail, force=True)
        self.finished = True
