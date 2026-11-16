"""`reachy-demo translate`: live translation of laptop mic or a WAV file to the terminal."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv

from reachy_demo.audio.sources import mic_chunks, wav_chunks
from reachy_demo.voice.translate import translate_stream


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("translate", help="live-translate laptop mic or a WAV file")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mic", action="store_true", help="laptop microphone (needs --extra mic)")
    src.add_argument("--file", type=Path, help="16-bit WAV")
    p.add_argument("--to", choices=["en", "vi"], default="en", help="target language")


async def render(deltas: AsyncIterator[str], out: TextIO) -> None:
    async for text in deltas:
        out.write(text)
        out.flush()
    out.write("\n")


def run(args: argparse.Namespace) -> int:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print("config error: OPENAI_API_KEY is required", file=sys.stderr)
        return 2
    if args.file is not None and not args.file.is_file():
        print(f"error: no such file: {args.file}", file=sys.stderr)
        return 2
    source = mic_chunks() if args.mic else wav_chunks(args.file)
    try:
        asyncio.run(render(translate_stream(source, api_key=key, target=args.to), sys.stdout))
    except KeyboardInterrupt:
        return 0
    except (ValueError, RuntimeError) as e:  # bad WAV format, mic extra missing
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ConnectionError as e:
        print(f"\ntranslation stopped: {e}", file=sys.stderr)
        return 1
    return 0
