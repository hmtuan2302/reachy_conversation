"""Mic diagnostic: shows your mic level live, saves 8 s to data/mic_check.wav, then transcribes it.

Run from the repo root:  uv run python scripts/mic_check.py
Speak normally during the 8 s countdown.
"""

import asyncio
import os
import sys
import wave
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv

from reachy_demo.audio.sources import RATE, mic_chunks, wav_chunks
from reachy_demo.voice.transcribe import transcribe_stream

OUT = Path("data/mic_check.wav")


async def record(seconds: float = 8.0) -> bytes:
    pcm = bytearray()
    async for chunk in mic_chunks():
        pcm += chunk
        x = np.frombuffer(chunk, dtype="<i2").astype(np.float64)
        rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
        peak = int(np.abs(x).max()) if len(x) else 0
        bar = "#" * min(60, int(rms / 100))
        left = seconds - len(pcm) / 2 / RATE
        print(f"{left:4.1f}s  rms {rms:7.0f}  peak {peak:6d}  {bar}", flush=True)
        if left <= 0:
            return bytes(pcm)
    return bytes(pcm)


async def main() -> int:
    load_dotenv(find_dotenv(usecwd=True))
    print("Speak now (8 s)...")
    pcm = await record()
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    print(f"\nsaved {OUT}  overall rms {np.sqrt(np.mean(x**2)):.0f}  peak {int(np.abs(x).max())}")
    print("Hint: speech is usually rms > 300 and peak > 3000. Near 0 means silence/wrong mic.\n")
    print("Transcribing the recording...")
    lines = [t async for t in transcribe_stream(wav_chunks(OUT), hf_token=os.environ["HF_TOKEN"])]
    print("\n".join(lines) if lines else "(no speech detected by the server)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
