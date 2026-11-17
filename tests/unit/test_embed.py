import sys
from pathlib import Path

import pytest

from reachy_demo.memory import embed


def test_load_embedder_off() -> None:
    assert embed.load_embedder({"EMBED": "off"}) == (None, "keywords only (EMBED=off)")


def test_load_embedder_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    e, status = embed.load_embedder({})
    assert e is None and "uv sync --extra embed" in status


def test_load_embedder_download_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import huggingface_hub

    def boom(*_a: object, **_k: object) -> str:
        raise OSError("401 gated repo")

    monkeypatch.chdir(tmp_path)  # no models/ folder here, so it must download
    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    e, status = embed.load_embedder({"HF_TOKEN": "hf_x"})
    assert e is None
    assert status.startswith("keywords only (model download failed") and "401" in status


def test_load_embedder_resumes_cut_off_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub

    calls: list[object] = []
    monkeypatch.chdir(tmp_path)
    graph = embed.model_dir(embed.DEFAULT_REPO) / embed.ONNX_FILE
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"graph only")  # model.onnx without model.onnx_data: an interrupted download
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: calls.append(a))
    embed.load_embedder({})
    assert calls  # downloaded again, not loaded half-finished


def test_load_embedder_bad_local_folder(tmp_path: Path) -> None:
    e, status = embed.load_embedder({"EMBED_MODEL": str(tmp_path)})
    assert e is None and status.startswith("keywords only (model load failed")


def test_local_dir_is_keyed_by_repo() -> None:
    assert embed.model_dir("onnx-community/embeddinggemma-300m-ONNX") == Path(
        "models/onnx-community--embeddinggemma-300m-ONNX"
    )


def test_model_key_follows_the_source() -> None:
    a = embed.model_key("onnx-community/embeddinggemma-300m-ONNX")
    b = embed.model_key("someone/other-embedder-ONNX")
    assert a == "onnx-community/embeddinggemma-300m-ONNX@256" and a != b
    assert embed.model_key("/models/my-export/") == "my-export@256"
