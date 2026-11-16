import subprocess
import sys
from pathlib import Path


def test_cli_check_config_end_to_end(tmp_path: Path) -> None:
    env = {"ROBOT_HOST": "10.0.0.2", "OPENAI_API_KEY": "sk-x", "LOG_DIR": str(tmp_path), "PATH": ""}
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "reachy_demo", "check-config"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert list(tmp_path.glob("run-*.jsonl"))


def test_cli_fails_cleanly_without_config(tmp_path: Path) -> None:
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "reachy_demo", "check-config"],
        env={"PATH": ""},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 2 and "ROBOT_HOST" in out.stderr
