"""`reachy-demo enroll`: save the owner's voiceprint from WAV recordings of their voice."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from reachy_demo.voice.speaker_id import (
    DEFAULT_MODEL,
    VOICEPRINT,
    Embed,
    load_sherpa_embed,
    normalize,
    read_wav,
    voiceprint,
)

MIN_CLIP_S = 3.0


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("enroll", help="save the owner's voiceprint from WAV files")
    p.add_argument("--file", type=Path, action="append", required=True, help="owner WAV (repeat)")
    p.add_argument("--test", type=Path, nargs="*", default=[], help="WAVs to score afterwards")
    p.add_argument("--out", type=Path, default=VOICEPRINT, help=f"default {VOICEPRINT}")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(normalize(a) @ normalize(b))


def run(args: argparse.Namespace, embed: Embed | None = None) -> int:
    clips: list[tuple[Path, np.ndarray, int]] = []
    for path in [*args.file, *args.test]:
        try:
            x, rate = read_wav(path)
        except (OSError, ValueError, EOFError) as e:
            print(f"error: cannot read {path}: {e}", file=sys.stderr)
            return 2
        if path in args.file and len(x) < MIN_CLIP_S * rate:
            print(f"error: {path} is {len(x) / rate:.1f} s, need at least 3 s", file=sys.stderr)
            return 2
        clips.append((path, x, rate))
    if embed is None:
        model = Path(os.environ.get("SPEAKER_MODEL", "").strip() or DEFAULT_MODEL)
        try:
            embed = load_sherpa_embed(model)
        except (ImportError, FileNotFoundError) as e:
            print(
                f"error: {e} (uv sync --extra speaker; model download: docs/usage/agent.md)",
                file=sys.stderr,
            )
            return 2
    owner = clips[: len(args.file)]
    vp = voiceprint([embed(x, rate) for _, x, rate in owner])
    _, x, rate = owner[0]
    half = len(x) // 2
    print(
        f"self-score: {_cos(embed(x[:half], rate), embed(x[half:], rate)):.2f}"
        "  (first half vs second half)"
    )
    for path, x, rate in clips[len(args.file) :]:
        print(f"test {path}: {_cos(embed(x, rate), vp):.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, vp)
    print(f"saved {args.out}")
    return 0
