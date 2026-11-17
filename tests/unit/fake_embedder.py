"""Deterministic stand-in for the embedding model: bag of words hashed into 256 dims."""

from __future__ import annotations

import re
import zlib
from typing import Literal

import numpy as np


class FakeEmbedder:
    model = "fake@256"

    def __init__(self, synonyms: dict[str, str] | None = None, fail: bool = False) -> None:
        self.synonyms = synonyms or {}
        self.fail = fail
        self.calls = 0

    def embed(self, texts: list[str], kind: Literal["query", "doc"]) -> np.ndarray:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedder broke")
        out = np.zeros((len(texts), 256), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, zlib.crc32(self.synonyms.get(w, w).encode()) % 256] += 1.0
            n = np.linalg.norm(out[i])
            out[i] = out[i] / n if n else out[i]
        return out
