import time
from pathlib import Path

import numpy as np
import pytest

from reachy_demo.voice.speaker_id import (
    DEFAULT_HIGH,
    DEFAULT_LOW,
    DEFAULT_MODEL,
    load_sherpa_embed,
    normalize,
    read_wav,
)

OWNER_1 = Path("data/my_voice1.wav")
OWNER_2 = Path("data/my_voice_2.wav")
OTHER = Path("data/girl-saying-hello_E_major.wav")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not all(p.is_file() for p in (DEFAULT_MODEL, OWNER_1, OWNER_2, OTHER)),
        reason="needs the CAM++ model and the owner's WAVs",
    ),
]


def test_enroll_scores_second_owner_clip_high() -> None:
    pytest.importorskip("sherpa_onnx")
    embed = load_sherpa_embed(DEFAULT_MODEL)
    vp = normalize(embed(*read_wav(OWNER_1)))
    owner = float(normalize(embed(*read_wav(OWNER_2))) @ vp)
    other = float(normalize(embed(*read_wav(OTHER))) @ vp)
    assert owner >= DEFAULT_HIGH and other < DEFAULT_LOW


def test_embed_is_fast() -> None:
    pytest.importorskip("sherpa_onnx")
    embed = load_sherpa_embed(DEFAULT_MODEL)
    x = np.zeros(3 * 24_000, dtype=np.float32) + 0.01
    embed(x, 24_000)  # warm-up
    t = time.perf_counter()
    embed(x, 24_000)
    assert time.perf_counter() - t < 0.2
