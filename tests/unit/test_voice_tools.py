import wave
from pathlib import Path

from reachy_demo.agent.tools.voice import SpeechBuffer, VoiceTools
from reachy_demo.audio.sources import RATE
from reachy_demo.memory.store import OWNER, Speaker

ONE_S = b"\x01\x00" * RATE


def test_buffer_keeps_only_recent_speech() -> None:
    buf = SpeechBuffer(keep_s=2)
    buf.feed(b"\x09\x00" * RATE, speaking=False)  # silence is not kept
    for _ in range(3):
        buf.feed(ONE_S, speaking=True)
    assert buf.seconds() == 2 and buf.pcm() == ONE_S * 2


async def test_clone_saves_wav_and_stop_deletes_it(tmp_path: Path) -> None:
    path = tmp_path / "voice" / "clone.wav"
    buf = SpeechBuffer()
    tools = VoiceTools(buf, speaker=lambda: OWNER, path=path)
    buf.feed(ONE_S, speaking=True)
    assert "need_more_speech" in (await tools.call("clone_voice", "{}"))["error"]
    assert not path.exists()
    for _ in range(3):
        buf.feed(ONE_S, speaking=True)
    assert await tools.call("clone_voice", "{}") == {"ok": True, "seconds": 4.0}
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == RATE and w.getnframes() == 4 * RATE
    assert await tools.call("stop_cloning_voice", "{}") == {"ok": True}
    assert not path.exists()


async def test_guest_cannot_clone(tmp_path: Path) -> None:
    buf = SpeechBuffer()
    for _ in range(4):
        buf.feed(ONE_S, speaking=True)
    tools = VoiceTools(buf, speaker=lambda: Speaker(2, "guest"), path=tmp_path / "c.wav")
    assert "not_owner" in (await tools.call("clone_voice", "{}"))["error"]
    assert not (tmp_path / "c.wav").exists()


async def test_fixed_sample_is_cloned_without_mic_speech(tmp_path: Path) -> None:
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFF-test")
    path = tmp_path / "voice" / "clone.wav"
    tools = VoiceTools(SpeechBuffer(), speaker=lambda: OWNER, path=path, sample=sample)
    assert (await tools.call("clone_voice", "{}"))["ok"] is True
    assert path.read_bytes() == b"RIFF-test"
    assert await tools.call("stop_cloning_voice", "{}") == {"ok": True}
    assert not path.exists() and sample.exists()  # stop never deletes the test file
    missing = VoiceTools(SpeechBuffer(), lambda: OWNER, path=path, sample=tmp_path / "no.wav")
    assert "cannot read CLONE_AUDIO" in (await missing.call("clone_voice", "{}"))["error"]
