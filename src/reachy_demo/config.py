"""Typed, validated configuration loaded from the environment (.env supported)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


class ConfigError(Exception):
    """Raised when required settings are missing or invalid."""


@dataclass(frozen=True)
class Config:
    robot_host: str
    openai_api_key: str = field(repr=False)
    realtime_model: str = ""
    realtime_voice: str = ""
    vision_model: str = ""
    gate_threshold: float = 0.5
    vad_threshold: float = 0.5
    vad_hangover_ms: int = 450
    onset_db: float = 15.0
    perception_max_hz: int = 10
    log_dir: Path = field(default_factory=lambda: Path("data/logs"))
    log_level: str = "INFO"
    record_audio: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        if env is None:
            load_dotenv(find_dotenv(usecwd=True))  # .env of the folder you run from
        src: Mapping[str, str] = os.environ if env is None else env
        errors: list[str] = []

        def req(key: str) -> str:
            value = src.get(key, "").strip()
            if not value:
                errors.append(f"{key} is required")
            return value

        def num[T: (int, float)](key: str, default: T) -> T:
            raw = src.get(key, "").strip()
            if not raw:
                return default
            try:
                return type(default)(raw)
            except ValueError:
                errors.append(f"{key} must be a {type(default).__name__}, got {raw!r}")
                return default

        cfg = cls(
            robot_host=req("ROBOT_HOST"),
            openai_api_key=req("OPENAI_API_KEY"),
            realtime_model=src.get("REALTIME_MODEL", ""),
            realtime_voice=src.get("REALTIME_VOICE", ""),
            vision_model=src.get("VISION_MODEL", ""),
            gate_threshold=num("GATE_THRESHOLD", 0.5),
            vad_threshold=num("VAD_THRESHOLD", 0.5),
            vad_hangover_ms=num("VAD_HANGOVER_MS", 450),
            onset_db=num("ONSET_DB", 15.0),
            perception_max_hz=num("PERCEPTION_MAX_HZ", 10),
            log_dir=Path(src.get("LOG_DIR", "data/logs")),
            log_level=src.get("LOG_LEVEL", "INFO").upper(),
            record_audio=src.get("RECORD_AUDIO", "false").lower() in {"1", "true", "yes"},
        )
        if errors:
            raise ConfigError("; ".join(errors))
        return cfg
