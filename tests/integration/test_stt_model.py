import time
from pathlib import Path

import numpy as np
import pytest

from reachy_demo.audio.sources import wav_chunks
from reachy_demo.voice import local_stt
from reachy_demo.voice.turns import SMART_TURN_SAMPLES

VI_WAV = Path("data/stt/vi.wav")
EN_WAV = Path("data/stt/en.wav")
VI_MARKS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")

pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def models() -> local_stt.LocalModels:
    pytest.importorskip("faster_whisper")
    return local_stt.load_models()


def test_smart_turn_features_match_transformers() -> None:
    pytest.importorskip("faster_whisper")
    transformers = pytest.importorskip("transformers")
    from faster_whisper.feature_extractor import FeatureExtractor

    audio = np.random.default_rng(0).standard_normal(3 * 16000).astype(np.float32) * 0.1
    padded = np.pad(audio, (SMART_TURN_SAMPLES - len(audio), 0))
    ours = local_stt.smart_turn_features(padded, FeatureExtractor(chunk_length=8))
    theirs = transformers.WhisperFeatureExtractor(chunk_length=8)(
        padded,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=SMART_TURN_SAMPLES,
        truncation=True,
        do_normalize=True,
    ).input_features
    assert ours.shape == theirs.shape == (1, 80, 800)
    assert np.allclose(ours, theirs, atol=1e-3)


def test_models_run_and_score_in_range(models: local_stt.LocalModels) -> None:
    silence = np.zeros(SMART_TURN_SAMPLES, np.float32)
    assert 0.0 <= models.turn(silence) <= 1.0
    assert 0.0 <= models.vad(np.zeros(512, np.float32)) <= 1.0
    start = time.perf_counter()
    models.turn(silence)
    print(f"smart turn: {(time.perf_counter() - start) * 1000:.0f} ms")


@pytest.mark.parametrize(
    ("wav", "vietnamese"),
    [
        pytest.param(w, vi, marks=pytest.mark.skipif(not w.is_file(), reason=f"record {w} first"))
        for w, vi in ((VI_WAV, True), (EN_WAV, False))
    ],
)
async def test_wav_transcribes_in_its_language(
    models: local_stt.LocalModels, wav: Path, vietnamese: bool
) -> None:
    lines = [t async for t in local_stt.transcribe_local(wav_chunks(wav), models=models)]
    print(lines)
    text = " ".join(lines).lower()
    assert text
    assert bool(VI_MARKS & set(text)) == vietnamese
