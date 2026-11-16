"""Structured JSONL logging with a per-turn trace id and secret redaction."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterable
from contextvars import ContextVar
from pathlib import Path

trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra= fields are merged in; secrets are masked."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, object] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if (tid := trace_id.get()) is not None:
            data["trace_id"] = tid
        data.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        line = json.dumps(data, default=str)
        for secret in self._secrets:
            line = line.replace(secret, "***")
        return line


def configure_logging(log_dir: Path, level: str = "INFO", secrets: Iterable[str] = ()) -> Path:
    """Console (human) + file (JSONL, data/logs/run-<ts>.jsonl). Returns the file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    secrets = list(secrets)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter(secrets))
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers[:] = [file_handler, console]
    root.setLevel(level)
    return path
