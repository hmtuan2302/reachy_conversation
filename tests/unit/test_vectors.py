import numpy as np

from reachy_demo.memory.store import OWNER_ID, MemoryStore
from reachy_demo.memory.vectors import VectorIndex, embed_one
from tests.unit.fake_embedder import FakeEmbedder


def _v(*hot: int) -> np.ndarray:
    v = np.zeros(256, np.float32)
    v[list(hot)] = 1.0
    return v / np.linalg.norm(v)


def test_top_ranks_by_cosine_within_allowed() -> None:
    idx = VectorIndex("m")
    idx.put("fact", 1, _v(1))
    idx.put("fact", 2, _v(1, 2))
    idx.put("fact", 3, _v(1, 2, 3))
    hits = idx.top(_v(1), "fact", 5, allowed={2, 3})
    assert [i for i, _ in hits] == [2, 3]
    assert hits[0][1] > hits[1][1]
    assert idx.top(_v(1), "fact", 5, allowed={1, 2, 3}, min_sim=0.9) == [(1, 1.0)]


def test_top_empty_index_and_empty_allowed_return_empty() -> None:
    idx = VectorIndex("m")
    assert idx.top(_v(1), "fact", 3, allowed={1}) == []
    idx.put("fact", 1, _v(1))
    assert idx.top(_v(1), "fact", 3, allowed=set()) == []
    assert idx.top(_v(1), "episode", 3, allowed={1}) == []


def test_put_overwrites_and_drop_removes() -> None:
    idx = VectorIndex("m")
    idx.put("fact", 1, _v(1))
    idx.put("fact", 1, _v(2))
    assert idx.top(_v(2), "fact", 3, allowed={1})[0][1] > 0.99
    idx.drop("fact", 1)
    idx.drop("fact", 99)
    assert idx.top(_v(2), "fact", 3, allowed={1}) == []


def test_load_ignores_other_models() -> None:
    s = MemoryStore(":memory:", lambda: 0.0)
    f = s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="tea")
    g = s.add_fact(subject_person_id=OWNER_ID, subject_text="Tuan", text="coffee")
    s.set_vec("fact", f, "old@768", np.ones(768, np.float32))
    s.set_vec("fact", g, "m", _v(4))
    idx = VectorIndex.load(s, "m")
    assert [i for i, _ in idx.top(_v(4), "fact", 3, allowed={f, g})] == [g]


def _cos(e: FakeEmbedder, a: str, b: str) -> float:
    va, vb = e.embed([a, b], "doc")
    return float(va @ vb)


def test_fake_embedder_synonyms_make_texts_similar() -> None:
    e = FakeEmbedder({"copilot": "aitool", "claude": "aitool"})
    assert _cos(e, "uses Copilot", "uses Claude") > 0.9
    assert _cos(e, "uses Copilot", "sister Mai visited") < 0.3


async def test_embed_one_returns_none_on_failure() -> None:
    assert await embed_one(FakeEmbedder(fail=True), "x", "query") is None
    assert await embed_one(None, "x", "query") is None
    v = await embed_one(FakeEmbedder(), "hello there", "query")
    assert v is not None and v.shape == (256,)


def test_gemma_rejects_short_vectors() -> None:
    import pytest

    from reachy_demo.memory.embed import GemmaEmbedder

    class Sess:
        def run(self, *_a: object) -> list[np.ndarray]:
            return [np.ones((1, 128), np.float32)]

    class Tok:
        def encode_batch(self, texts: list[str]) -> list[object]:
            class E:
                ids = [1]
                attention_mask = [1]

            return [E() for _ in texts]

    g = GemmaEmbedder.__new__(GemmaEmbedder)
    g._tok, g._sess = Tok(), Sess()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="256"):
        g.embed(["x"], "doc")
