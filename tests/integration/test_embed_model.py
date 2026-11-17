from pathlib import Path

import pytest

from reachy_demo.memory import embed

DIR = embed.model_dir(embed.DEFAULT_REPO)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (DIR / "onnx" / "model.onnx").exists(), reason="model not downloaded"),
]


def test_gemma_vn_en_pair_is_close() -> None:
    e = embed.GemmaEmbedder(Path(DIR))
    vn, en, other = e.embed(
        ["Tôi có con chó tên là Bin", "I have a dog named Bin", "I live in Saigon"], "doc"
    )
    assert vn.shape == (256,)
    assert abs(float(vn @ vn) - 1.0) < 1e-4
    assert float(vn @ en) > float(vn @ other)
