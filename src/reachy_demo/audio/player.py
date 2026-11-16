"""Play streamed PCM through a player subprocess, then save it as a WAV."""

from __future__ import annotations

import asyncio
import shlex
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path

RATE = 24_000
# Windows-native playback from WSL (same route as the mic); raw PCM on stdin.
DEFAULT_PLAYER = (
    "ffplay.exe -autoexit -nodisp -loglevel error -f s16le -ar 24000 -ch_layout mono -i -"
)


class PlayerError(Exception):
    """Playback failed. The audio was still saved."""


def speech_path(directory: Path, n: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{n}.wav"


def _save(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


async def play_and_save(
    chunks: AsyncIterator[bytes], *, path: Path, player_cmd: str | None
) -> Path:
    """Play chunks as they arrive (if `player_cmd`), and always save everything received to `path`.

    Source errors are re-raised after saving; playback problems raise PlayerError after saving.
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
    try:
        async for chunk in chunks:
            pcm += chunk
            if proc and proc.stdin and not problem:
                try:
                    proc.stdin.write(chunk)
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    problem = "player exited before the audio ended"
        finished = True
    finally:
        _save(path, bytes(pcm))
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


async def _finish(proc: asyncio.subprocess.Process) -> None:
    """Close the player's input and wait until it has played everything."""
    if proc.stdin:
        try:
            proc.stdin.close()
            await proc.stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass
    await proc.wait()
