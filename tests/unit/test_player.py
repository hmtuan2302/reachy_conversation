import asyncio
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from reachy_demo.audio.player import PlayerError, play_and_save, speech_path


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for p in parts:
        yield p


def _wav_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        return w.readframes(w.getnframes())


async def test_player_gets_every_byte_and_wav_matches(tmp_path: Path) -> None:
    played = tmp_path / "played.raw"
    out = tmp_path / "a.wav"
    cmd = f"sh -c 'cat > {played}'"
    assert await play_and_save(_chunks(b"\x01\x00", b"\x02\x00"), path=out, player_cmd=cmd) == out
    assert played.read_bytes() == b"\x01\x00\x02\x00"
    assert _wav_pcm(out) == b"\x01\x00\x02\x00"


async def test_no_player_saves_only(tmp_path: Path) -> None:
    out = tmp_path / "a.wav"
    await play_and_save(_chunks(b"\x05\x00"), path=out, player_cmd=None)
    assert _wav_pcm(out) == b"\x05\x00"


async def test_missing_player_still_saves(tmp_path: Path) -> None:
    out = tmp_path / "a.wav"
    with pytest.raises(PlayerError, match="no-such-player"):
        await play_and_save(_chunks(b"\x05\x00"), path=out, player_cmd="no-such-player -i -")
    assert _wav_pcm(out) == b"\x05\x00"


async def test_player_exiting_early_still_saves(tmp_path: Path) -> None:
    out = tmp_path / "a.wav"
    big = b"\x00\x00" * 200_000  # more than a pipe buffer, so writes hit the closed pipe
    with pytest.raises(PlayerError, match="exited"):
        await play_and_save(_chunks(big, big), path=out, player_cmd="sh -c 'exit 3'")
    assert len(_wav_pcm(out)) == 2 * len(big)


async def test_stream_error_saves_partial_audio(tmp_path: Path) -> None:
    async def broken() -> AsyncIterator[bytes]:
        yield b"\x07\x00"
        raise RuntimeError("stream broke")

    out = tmp_path / "a.wav"
    with pytest.raises(RuntimeError, match="stream broke"):
        await play_and_save(broken(), path=out, player_cmd=None)
    assert _wav_pcm(out) == b"\x07\x00"


def test_speech_path_is_timestamped_and_numbered(tmp_path: Path) -> None:
    p = speech_path(tmp_path / "speech", 3)
    assert p.parent == tmp_path / "speech" and p.parent.is_dir()
    assert p.name.endswith("-3.wav") and len(p.name) == len("20261116-091502-3.wav")


async def test_gain_scales_played_audio_not_wav(tmp_path: Path) -> None:
    played, out = tmp_path / "played.raw", tmp_path / "a.wav"
    pcm = np.full(2400, 4000, "<i2").tobytes()
    cmd = f"sh -c 'cat > {played}'"
    await play_and_save(_chunks(pcm), path=out, player_cmd=cmd, gain=lambda: 0.25)
    assert set(np.frombuffer(played.read_bytes(), "<i2").tolist()) == {1000}
    assert _wav_pcm(out) == pcm


async def test_playback_is_paced(tmp_path: Path) -> None:
    played, out = tmp_path / "played.raw", tmp_path / "a.wav"
    pcm = b"\x00\x00" * (24000 // 2)  # 0.5 s arriving at once
    t0 = time.monotonic()
    await play_and_save(_chunks(pcm), path=out, player_cmd=f"sh -c 'cat > {played}'", lead_s=0.1)
    assert (
        time.monotonic() - t0 >= 0.28
    )  # last 100 ms piece is written 0.1 s (lead) before it plays
    assert played.read_bytes() == pcm


async def test_cancel_kills_player_and_saves(tmp_path: Path) -> None:
    out = tmp_path / "a.wav"
    first = b"\x01\x00" * 2400

    async def endless() -> AsyncIterator[bytes]:
        yield first
        await asyncio.sleep(60)
        yield b"\x02\x00"

    task = asyncio.create_task(
        play_and_save(endless(), path=out, player_cmd="sh -c 'cat >/dev/null'")
    )
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert _wav_pcm(out) == first


async def test_pacing_starts_when_audio_arrives(tmp_path: Path) -> None:
    played, out = tmp_path / "played.raw", tmp_path / "a.wav"
    pcm = b"\x00\x00" * (24000 // 2)  # 0.5 s, arriving at once after a 0.5 s TTS delay

    async def late() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.5)
        yield pcm

    t0 = time.monotonic()
    await play_and_save(late(), path=out, player_cmd=f"sh -c 'cat > {played}'", lead_s=0.1)
    assert time.monotonic() - t0 >= 0.5 + 0.28  # the wait for audio is not counted as played


async def test_on_play_reports_when_the_speaker_runs_out(tmp_path: Path) -> None:
    ends: list[float] = []
    await play_and_save(
        _chunks(b"\x00\x00" * 2400),
        path=None,
        player_cmd="sh -c 'cat > /dev/null'",
        on_play=ends.append,
    )
    assert ends and ends[-1] > 0
