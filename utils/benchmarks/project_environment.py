"""Small dependency-free reader for project-level benchmark defaults."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Mapping, Sequence


_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse shell-compatible KEY=VALUE lines without executing the file."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        key = key.rstrip()
        if not separator or not _KEY.fullmatch(key):
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        tokens = shlex.split(raw_value, comments=True, posix=True)
        if len(tokens) > 1:
            raise ValueError(f"{path}:{line_number}: invalid environment value")
        values[key] = tokens[0] if tokens else ""
    return values


def configured_backend(
    project_root: Path,
    *,
    default: str,
    allowed: Sequence[str],
    environment: Mapping[str, str] | None = None,
) -> str:
    """Return process env > project .env > default and validate the result."""
    active_environment = os.environ if environment is None else environment
    value = active_environment.get("LLM_LAB_BACKEND")
    if not value:
        configured_path = active_environment.get("LLM_LAB_ENV_FILE")
        env_path = Path(configured_path) if configured_path else project_root / ".env"
        if configured_path and not env_path.is_file():
            raise ValueError(f"environment file does not exist: {env_path}")
        value = read_dotenv(env_path).get("LLM_LAB_BACKEND", default)
    if value not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(f"invalid LLM_LAB_BACKEND: {value} (expected {choices})")
    return value
