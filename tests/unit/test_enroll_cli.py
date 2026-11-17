import argparse
import wave
from pathlib import Path

import numpy as np
import pytest

from reachy_demo.voice import enroll_cli


def _wav(path: Path, seconds: float, rate: int = 16_000) -> Path:
    x = (np.sin(np.arange(int(seconds * rate)) / 7) * 8000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return path


def fake_embed(x: np.ndarray, rate: int) -> np.ndarray:
    return np.array([3.0, float(len(x) % 5)])


def _args(files: list[Path], out: Path, test: list[Path] | None = None) -> argparse.Namespace:
    return argparse.Namespace(file=files, test=test or [], out=out)


def test_enroll_writes_voiceprint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "voice" / "owner.npy"
    files = [_wav(tmp_path / "a.wav", 4), _wav(tmp_path / "b.wav", 5, rate=48_000)]
    assert enroll_cli.run(_args(files, out), embed=fake_embed) == 0
    assert np.linalg.norm(np.load(out)) == pytest.approx(1.0)
    assert "self-score:" in capsys.readouterr().out


def test_enroll_rejects_short_clip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "owner.npy"
    assert enroll_cli.run(_args([_wav(tmp_path / "a.wav", 2)], out), embed=fake_embed) == 2
    assert not out.exists()
    assert "at least 3 s" in capsys.readouterr().err


def test_enroll_rejects_missing_file(tmp_path: Path) -> None:
    out = tmp_path / "owner.npy"
    assert enroll_cli.run(_args([tmp_path / "nope.wav"], out), embed=fake_embed) == 2


def test_enroll_prints_test_scores(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    files = [_wav(tmp_path / "a.wav", 4)]
    tests = [_wav(tmp_path / "x.wav", 3), _wav(tmp_path / "y.wav", 3)]
    assert enroll_cli.run(_args(files, tmp_path / "o.npy", tests), embed=fake_embed) == 0
    assert capsys.readouterr().out.count("test ") == 2
