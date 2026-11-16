from pathlib import Path

import pytest

from reachy_demo.config import Config, ConfigError

BASE = {"ROBOT_HOST": "10.0.0.2", "OPENAI_API_KEY": "sk-secret"}


def test_defaults_and_overrides() -> None:
    cfg = Config.from_env({**BASE, "VAD_HANGOVER_MS": "600", "RECORD_AUDIO": "true"})
    assert cfg.robot_host == "10.0.0.2"
    assert cfg.vad_hangover_ms == 600
    assert cfg.record_audio is True
    assert cfg.log_dir == Path("data/logs")


def test_missing_required_reports_all() -> None:
    with pytest.raises(ConfigError) as exc:
        Config.from_env({})
    assert "ROBOT_HOST" in str(exc.value) and "OPENAI_API_KEY" in str(exc.value)


def test_bad_number_is_an_error() -> None:
    with pytest.raises(ConfigError, match="ONSET_DB"):
        Config.from_env({**BASE, "ONSET_DB": "loud"})


def test_api_key_not_in_repr() -> None:
    assert "sk-secret" not in repr(Config.from_env(BASE))
