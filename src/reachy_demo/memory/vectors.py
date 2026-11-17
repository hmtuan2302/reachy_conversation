"""Vectors for memory search: the embedder interface and an in-RAM cosine index.

Spec: docs/superpowers/specs/2026-11-17-smart-context-design.md §5–§7.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from reachy_demo.memory.store import VecKind

if TYPE_CHECKING:
    from reachy_demo.memory.store import MemoryStore

log = logging.getLogger(__name__)

EmbedKind = Literal["query", "doc"]


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str], kind: EmbedKind) -> np.ndarray:
        """(n, 256) float32, rows L2-normalized."""
        ...


async def embed_one(embedder: Embedder | None, text: str, kind: EmbedKind) -> np.ndarray | None:
    """One vector off the event loop, or None (no embedder, or it failed: callers use keywords)."""
    if embedder is None:
        return None
    try:
        vecs = await asyncio.to_thread(embedder.embed, [text], kind)
    except Exception:  # fail small: a broken model degrades search, never the turn
        log.warning("embedding failed; using keyword search", exc_info=True)
        return None
    return np.asarray(vecs[0], dtype=np.float32)


class VectorIndex:
    """Vectors of one model, per kind, searched with one matrix multiply."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._vecs: dict[VecKind, dict[int, np.ndarray]] = {"fact": {}, "episode": {}}
        self._cache: dict[VecKind, tuple[np.ndarray, np.ndarray] | None] = {
            "fact": None,
            "episode": None,
        }

    @classmethod
    def load(cls, store: MemoryStore, model: str) -> VectorIndex:
        idx = cls(model)
        for kind in ("fact", "episode"):
            for item_id, vec in store.load_vecs(kind, model):
                idx.put(kind, item_id, vec)
        return idx

    def put(self, kind: VecKind, item_id: int, vec: np.ndarray) -> None:
        self._vecs[kind][item_id] = np.asarray(vec, dtype=np.float32)
        self._cache[kind] = None

    def drop(self, kind: VecKind, item_id: int) -> None:
        if self._vecs[kind].pop(item_id, None) is not None:
            self._cache[kind] = None

    def top(
        self,
        query: np.ndarray,
        kind: VecKind,
        n: int,
        allowed: set[int],
        min_sim: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Highest cosine first, only among `allowed` ids (masked before ranking)."""
        if not allowed or not self._vecs[kind]:
            return []
        if self._cache[kind] is None:
            ids = np.fromiter(self._vecs[kind].keys(), dtype=np.int64)
            self._cache[kind] = (ids, np.stack(list(self._vecs[kind].values())))
        cached = self._cache[kind]
        assert cached is not None
        ids, mat = cached
        mask = np.isin(ids, np.fromiter(allowed, dtype=np.int64))
        if not mask.any():
            return []
        ids, sims = ids[mask], mat[mask] @ np.asarray(query, dtype=np.float32)
        order = np.argsort(-sims)[:n]
        return [(int(ids[i]), round(float(sims[i]), 6)) for i in order if sims[i] >= min_sim]


BACKFILL_BATCH = 16


async def store_vec(
    store: MemoryStore,
    index: VectorIndex | None,
    embedder: Embedder | None,
    kind: VecKind,
    item_id: int,
    text: str,
    vec: np.ndarray | None = None,
) -> None:
    """Embed (unless `vec` is given) and save one item; on failure it stays keyword-only."""
    if index is None:
        return
    index.drop(kind, item_id)  # never leave a vector of older text behind
    if vec is None:
        vec = await embed_one(embedder, text, "doc")
    if vec is not None:
        store.set_vec(kind, item_id, index.model, vec)
        index.put(kind, item_id, vec)


async def backfill(
    store: MemoryStore, embedder: Embedder, index: VectorIndex, limit: int
) -> tuple[int, int]:
    """Embed up to `limit` rows that have no vector for this model: (facts, episodes) done."""
    done = {"fact": 0, "episode": 0}
    for kind in ("fact", "episode"):
        rows = store.missing_vecs(kind, embedder.model, limit - sum(done.values()))
        for i in range(0, len(rows), BACKFILL_BATCH):
            batch = rows[i : i + BACKFILL_BATCH]
            try:
                vecs = await asyncio.to_thread(embedder.embed, [t for _, t in batch], "doc")
            except Exception:
                log.warning("memory backfill: embedding failed, will retry later", exc_info=True)
                return done["fact"], done["episode"]
            for (item_id, _), vec in zip(batch, vecs, strict=True):
                store.set_vec(kind, item_id, embedder.model, vec)
                index.put(kind, item_id, vec)
            done[kind] += len(batch)
    return done["fact"], done["episode"]
