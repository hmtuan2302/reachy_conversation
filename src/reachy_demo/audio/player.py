"""Play streamed PCM through a player subprocess, then save it as a WAV."""

from __future__ import annotations

import asyncio
import shlex
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import numpy as np

from reachy_demo.audio.sources import RATE, save_wav

BYTES_PER_S = RATE * 2
PIECE = BYTES_PER_S // 10  # 100 ms: how often the gain is re-read

# Windows-native playback from WSL (same route as the mic); raw PCM on stdin.
DEFAULT_PLAYER = (
    "ffplay.exe -autoexit -nodisp -loglevel error -f s16le -ar 24000 -ch_layout mono -i -"
)


class PlayerError(Exception):
    """Playback failed. The audio was still saved."""


def speech_path(directory: Path, n: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{n}.wav"


async def play_and_save(
    chunks: AsyncIterator[bytes],
    *,
    path: Path,
    player_cmd: str | None,
    gain: Callable[[], float] | None = None,
    lead_s: float = 0.2,
) -> Path:
    """Play chunks as they arrive (if `player_cmd`), and always save everything received to `path`.

    Playback is paced to stay only `lead_s` ahead of the speaker, so `gain()` (read per 100 ms
    piece) changes what is heard soon. The saved WAV is unscaled. Source errors are re-raised
    after saving; playback problems raise PlayerError after saving.
    """
    pcm = bytearray()
    problem = ""
    proc: asyncio.subprocess.Process | None = None
    if player_cmd:
        try:
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(player_cmd),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as e:
            problem = f"cannot start player {player_cmd.split()[0]!r}: {e}"
    finished = False
    plays_until = 0.0  # monotonic time the speaker runs out of audio already written
    try:
        async for chunk in chunks:
            pcm += chunk
            if not (proc and proc.stdin):
                continue
            for i in range(0, len(chunk), PIECE):
                if problem:
                    break
                if (ahead := plays_until - time.monotonic()) > lead_s:
                    await asyncio.sleep(ahead - lead_s)
                piece = chunk[i : i + PIECE]
                # a starved speaker (waiting for TTS) doesn't count as played
                plays_until = max(plays_until, time.monotonic()) + len(piece) / BYTES_PER_S
                try:
                    proc.stdin.write(_scaled(piece, gain() if gain else 1.0))
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    problem = "player exited before the audio ended"
        finished = True
    finally:
        save_wav(path, bytes(pcm))
        if proc:
            if finished:
                await _finish(proc)
            else:  # source failed or Ctrl-C: stop playback now
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()
    if proc and proc.returncode and not problem:
        problem = f"player exited with code {proc.returncode}"
    if problem:
        raise PlayerError(f"{problem} (audio saved to {path})")
    return path


def _scaled(piece: bytes, gain: float) -> bytes:
    if gain == 1.0:
        return piece
    x = np.frombuffer(piece[: len(piece) // 2 * 2], dtype="<i2").astype(np.float64) * gain
    return np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes()


async def _finish(proc: asyncio.subprocess.Process) -> None:
    """Close the player's input and wait until it has played everything."""
    if proc.stdin:
        try:
            proc.stdin.close()
            await proc.stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass
    await proc.wait()
