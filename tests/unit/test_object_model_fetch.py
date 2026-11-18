from pathlib import Path

import pytest

from reachy_demo.perception import objects
from reachy_demo.perception.objects import DEFAULT_MODEL, fetch_model, model_path


def test_short_names_map_to_the_yolox_files() -> None:
    assert model_path("") == DEFAULT_MODEL
    assert model_path(" Tiny ") == Path("models/yolox/yolox_tiny.onnx")
    assert model_path("my/own.onnx") == Path("my/own.onnx")


def test_a_missing_yolox_model_is_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    got: list[str] = []

    def fake(url: str, to: Path) -> None:
        got.append(url)
        Path(to).write_bytes(b"onnx")

    monkeypatch.setattr("urllib.request.urlretrieve", fake)
    model = tmp_path / "yolox" / "yolox_tiny.onnx"
    fetch_model(model)
    assert model.read_bytes() == b"onnx"
    assert got == [f"{objects.RELEASE_URL}/yolox_tiny.onnx"]
    fetch_model(model)  # there now: no second download
    assert len(got) == 1


def test_a_failed_download_leaves_no_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(url: str, to: Path) -> None:
        Path(to).write_bytes(b"half")
        raise OSError("connection reset")

    monkeypatch.setattr("urllib.request.urlretrieve", broken)
    model = tmp_path / "yolox_s.onnx"
    with pytest.raises(FileNotFoundError, match="download failed"):
        fetch_model(model)
    assert list(tmp_path.iterdir()) == []


def test_an_unknown_missing_file_is_not_downloaded(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nano, tiny"):
        fetch_model(tmp_path / "custom.onnx")
