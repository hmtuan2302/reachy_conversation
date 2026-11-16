"""`reachy-demo transcribe`: live speech-to-text of laptop mic or a WAV file."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from reachy_demo.audio.sources import Recording, mic_chunks, stdin_chunks, wav_chunks
from reachy_demo.voice.transcribe import transcribe_stream


def add_source_args(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mic", action="store_true", help="laptop microphone (needs --extra mic)")
    src.add_argument("--file", type=Path, help="16-bit PCM WAV")
    src.add_argument(
        "--stdin",
        action="store_true",
        help="raw 24 kHz mono s16le PCM piped in (e.g. from Windows ffmpeg.exe)",
    )


def source_from_args(args: argparse.Namespace) -> AsyncIterator[bytes]:
    """The audio source picked on the command line. ValueError if --file doesn't exist."""
    if args.stdin:
        return stdin_chunks(sys.stdin.buffer)
    if args.mic:
        return mic_chunks()
    if not args.file.is_file():
        raise ValueError(f"no such file: {args.file}")
    return wav_chunks(args.file)


RECORDINGS_DIR = Path("data/recordings")


def recorded(
    args: argparse.Namespace, source: AsyncIterator[bytes]
) -> tuple[AsyncIterator[bytes], Recording | None]:
    """Keep a copy of live input (--mic/--stdin) for debugging; a --file is already on disk."""
    if args.file is not None:
        return source, None
    rec = Recording()
    return rec.tee(source), rec


def save_recording(args: argparse.Namespace, rec: Recording | None) -> None:
    if rec and (path := rec.save(RECORDINGS_DIR, "mic" if args.mic else "stdin")):
        print(f"recording saved: {path}", file=sys.stderr)


def run_until_done(main: Coroutine[Any, Any, None]) -> int:
    """Run a pipeline and map its failures to exit codes (0 ok/Ctrl-C, 1 network, 2 input)."""
    try:
        asyncio.run(main)
    except KeyboardInterrupt:
        return 0
    except (ValueError, RuntimeError) as e:  # bad WAV format, mic missing or refused
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ConnectionError as e:
        print(f"transcription stopped: {e}", file=sys.stderr)
        return 1
    return 0


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("transcribe", help="live speech-to-text of laptop mic or a WAV file")
    add_source_args(p)


async def _print_lines(lines: AsyncIterator[str]) -> None:
    async for line in lines:
        print(line, flush=True)


def run(args: argparse.Namespace) -> int:
    load_dotenv(find_dotenv(usecwd=True))  # .env of the folder you run from
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("config error: HF_TOKEN is required", file=sys.stderr)
        return 2
    try:
        source = source_from_args(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    source, rec = recorded(args, source)
    code = run_until_done(_print_lines(transcribe_stream(source, hf_token=token)))
    save_recording(args, rec)  # also after Ctrl-C or an error: that's when it's most useful
    return code
