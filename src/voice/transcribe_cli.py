"""`reachy-demo transcribe`: live speech-to-text of laptop mic or a WAV file."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from reachy_demo.audio.sources import mic_chunks, wav_chunks
from reachy_demo.voice.transcribe import transcribe_stream


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("transcribe", help="live speech-to-text of laptop mic or a WAV file")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mic", action="store_true", help="laptop microphone (needs --extra mic)")
    src.add_argument("--file", type=Path, help="16-bit PCM WAV")


async def _print_lines(lines: AsyncIterator[str]) -> None:
    async for line in lines:
        print(line, flush=True)


def run(args: argparse.Namespace) -> int:
    load_dotenv(find_dotenv(usecwd=True))  # .env of the folder you run from
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("config error: HF_TOKEN is required", file=sys.stderr)
        return 2
    if args.file is not None and not args.file.is_file():
        print(f"error: no such file: {args.file}", file=sys.stderr)
        return 2
    source = mic_chunks() if args.mic else wav_chunks(args.file)
    try:
        asyncio.run(_print_lines(transcribe_stream(source, hf_token=token)))
    except KeyboardInterrupt:
        return 0
    except (ValueError, RuntimeError) as e:  # bad WAV format, mic missing or refused
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ConnectionError as e:
        print(f"transcription stopped: {e}", file=sys.stderr)
        return 1
    return 0
