"""embeddinggemma-300m through ONNX Runtime (no torch), and the env loader.

Spec: docs/superpowers/specs/2026-11-17-smart-context-design.md §7.1, spike results §13.
`tokenizers` and `huggingface_hub` come from the `embed` extra, so they are imported lazily.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from reachy_demo.memory.vectors import Embedder, EmbedKind

log = logging.getLogger(__name__)

DEFAULT_REPO = "onnx-community/embeddinggemma-300m-ONNX"
DIMS = 256
MAX_TOKENS = 256
ONNX_FILE = "onnx/model.onnx"  # fp32: the int8 file is ~5x slower on the laptop CPU (spike)
FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", ONNX_FILE + "*"]
PREFIX: dict[EmbedKind, str] = {
    "query": "task: search result | query: ",
    "doc": "title: none | text: ",
}


def model_key(source: str) -> str:
    """The key vectors are stored under: a different model or folder never mixes with these."""
    local = Path(source).is_absolute() or source.startswith(".")
    return f"{Path(source).name if local else source.strip('/')}@{DIMS}"


def model_dir(repo: str) -> Path:
    # Real files, not the HF cache: onnxruntime rejects external weights behind cache symlinks.
    return Path("models") / repo.replace("/", "--")


class GemmaEmbedder:
    def __init__(self, path: Path, model: str | None = None) -> None:
        self.model = model or model_key(str(path.resolve()))
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._tok.enable_truncation(MAX_TOKENS)
        self._tok.enable_padding()
        self._sess: Any = ort.InferenceSession(
            str(path / ONNX_FILE), providers=["CPUExecutionProvider"]
        )
        self.embed(["warm up"], "query")  # so the first real turn is not slow

    def embed(self, texts: list[str], kind: EmbedKind) -> np.ndarray:
        enc = self._tok.encode_batch([PREFIX[kind] + t for t in texts])
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        (vecs,) = self._sess.run(["sentence_embedding"], feed)
        if vecs.shape[1] < DIMS:  # another model: never mix sizes in one index
            raise ValueError(f"embedding has {vecs.shape[1]} dims, need at least {DIMS}")
        vecs = np.asarray(vecs[:, :DIMS], dtype=np.float32)  # Matryoshka: first dims, renormalize
        out: np.ndarray = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return out


def load_embedder(env: Mapping[str, str]) -> tuple[Embedder | None, str]:
    """The embedder and a one-line status for the console; None means keyword-only search."""
    if env.get("EMBED", "").strip().lower() in ("off", "0", "false", "no"):
        return None, "keywords only (EMBED=off)"
    try:
        import huggingface_hub
        import tokenizers  # noqa: F401
    except ImportError:
        return None, "keywords only (install: uv sync --extra embed)"
    source = env.get("EMBED_MODEL", "").strip() or DEFAULT_REPO
    path = Path(source) if Path(source).is_dir() else model_dir(source)
    # No weights file means a cut-off download; snapshot_download resumes it.
    if not (path / (ONNX_FILE + "_data")).exists() and not Path(source).is_dir():
        try:
            huggingface_hub.snapshot_download(
                source,
                local_dir=path,
                allow_patterns=FILES,
                token=env.get("HF_TOKEN", "").strip() or None,
            )
        except Exception as e:  # gated, offline, wrong name: memory still works with keywords
            log.warning("embedding model download failed", exc_info=True)
            return None, f"keywords only (model download failed: {e})"
    try:
        embedder = GemmaEmbedder(path, model_key(source))
    except Exception as e:
        log.warning("embedding model load failed", exc_info=True)
        return None, f"keywords only (model load failed: {e})"
    return embedder, "hybrid (embeddinggemma-300m)"
