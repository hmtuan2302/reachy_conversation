"""Record raw episodes for the VLA tracking dataset.

Faces (docs/usage/track-record-face.md) or objects (docs/usage/track-record.md). Run from the
repo root, with the agent app stopped (one owner per head):
    uv run python scripts/track_record.py --target face --targets data/track_face/people.json \\
        --lighting daylight --background plain_wall

Each episode: the scenario is printed, you press Enter, it records --seconds (Enter again stops
early) while the person or object moves as told. Then: Enter = record the next one,
r = other scenario, d = delete the episode just recorded, n TEXT = note for the next episode,
q = quit.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import logging
import os
import random
import shutil
import string
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from reachy_demo.agent.cli import FACE_TRACK_SCORE, camera_lag_from_env
from reachy_demo.attention.follow import Follow
from reachy_demo.attention.hints import chasing, with_body
from reachy_demo.perception.objects import (
    BoxesDetector,
    ObjectDetector,
    ObjectSource,
    PersonHeads,
    model_path,
)
from reachy_demo.robot.adapter import Robot, connect_robot, scored_face_detector
from reachy_demo.robot.motion import Motion
from reachy_demo.vla.record import (
    BACKGROUND,
    FACE_BACKGROUND,
    FACE_FACTORS,
    FACE_RAW_DIR,
    FACTORS,
    LIGHTING,
    RAW_DIR,
    VIDEO_FPS,
    EpisodeWriter,
    jolted,
    load_objects,
    next_episode_dir,
    next_scenario,
    recorded,
    sweep_move,
)

POSE_HZ = 50.0


def open_video(path: Path, w: int, h: int) -> Any:
    import cv2

    video = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), VIDEO_FPS, (w, h))
    if not video.isOpened():
        raise OSError(f"cannot write video {path}")
    return video


def session_name(raw: Path) -> str:
    """session_<today>_a, _b, ... : the first letter not used yet today."""
    day = datetime.date.today().isoformat()
    for c in string.ascii_lowercase:
        if not (raw / f"session_{day}_{c}").exists():
            return f"session_{day}_{c}"
    raise RuntimeError("26 sessions today already")


def record_threads(
    robot: Robot, motion: Motion, ep: EpisodeWriter, stop: threading.Event
) -> list[threading.Thread]:
    """Camera thread (every new frame, with its arrival time) and pose thread (50 Hz)."""

    def frames() -> None:
        last = None
        while not stop.is_set():
            img = robot.grab_frame()
            if img is None or img is last:  # LatestFrame hands out the same array until a new one
                time.sleep(
                    0.003
                )  # shortcut: arrival time ±3 ms; use the SDK's frame time if it gets one
                continue
            last = img
            ep.frame(img, time.monotonic())

    def poses() -> None:
        while not stop.wait(1 / POSE_HZ):
            t = time.monotonic()
            try:
                p = robot.read_pose()
            except Exception:  # noqa: BLE001, S112 - no reading yet: skip this tick
                continue
            cmd = motion.pose  # last pose sent; read only, the loop replaces the dict
            ep.pose(
                {"t": t, **p, **{f"cmd_{k}": cmd[k] for k in ("yaw", "pitch", "roll", "body_yaw")}}
            )

    threads = [threading.Thread(target=f, daemon=True) for f in (frames, poses)]
    for th in threads:
        th.start()
    return threads


async def main(args: argparse.Namespace) -> int:
    load_dotenv(find_dotenv(usecwd=True))
    face = args.target == "face"
    factors = FACE_FACTORS if face else FACTORS
    backgrounds = FACE_BACKGROUND if face else BACKGROUND
    if args.background not in backgrounds:
        print(f"--background for {args.target}: one of {', '.join(backgrounds)}")
        return 2
    args.raw = args.raw or (FACE_RAW_DIR if face else RAW_DIR)
    objects = load_objects(args.targets, face)
    lag_s = camera_lag_from_env()
    if lag_s is None:
        return 2
    session_dir = args.raw / (args.session or session_name(args.raw))
    rng = random.Random()  # noqa: S311 - scenario sampling, not security
    lines: asyncio.Future[str] | None = None

    async def line() -> str:
        """The next line typed (one reader for the whole run)."""
        nonlocal lines
        if lines is None:
            lines = asyncio.ensure_future(asyncio.to_thread(input))
        try:
            return (await asyncio.shield(lines)).strip()
        finally:
            if lines.done():
                lines = None

    with connect_robot(os.environ["ROBOT_HOST"]) as robot:
        if not robot.has_camera:
            print("no camera")
            return 1
        motion = Motion(robot.set_pose, on_error=lambda e: print(f"  (head error: {e})"))
        detector = (
            (lambda: BoxesDetector("face", scored_face_detector(FACE_TRACK_SCORE)))
            if face  # the agent's face tracking pipeline: YuNet on the laptop
            else partial(ObjectDetector, model_path(os.environ.get("OBJECT_MODEL", "")))
        )
        seen = ObjectSource(robot.grab_frame, detector, lag_s=lag_s)
        read = seen.read
        if face:  # as the agent's "track me": a body's head, or chase a face that just left
            heads = PersonHeads(
                robot.grab_frame,
                partial(ObjectDetector, model_path(os.environ.get("OBJECT_MODEL", ""))),
                lag_s=lag_s,
            )
            read = chasing(with_body(seen.read, heads.read))
        follow = Follow(motion, jolted(read, rng), seen.switch, poll_s=0.05, name="the target")
        print(f"session: {session_dir}")
        last: Path | None = None
        note = ""
        try:
            scenario = next_scenario(objects, recorded(args.raw), rng, args.held_out, factors)
            while True:
                print(
                    f"\nnext: {describe(scenario, factors)}" + (f"  [note: {note}]" if note else "")
                )
                print(
                    "Enter = record, r = other scenario, d = delete last, n TEXT = note, q = quit"
                )
                cmd = await line()
                if cmd == "q":
                    break
                if cmd == "r":
                    scenario = next_scenario(
                        objects, recorded(args.raw), rng, args.held_out, factors
                    )
                    continue
                if cmd.startswith("n "):
                    note = cmd[2:].strip()
                    continue
                if cmd == "d":
                    if last is not None:
                        shutil.rmtree(last)
                        print(f"deleted {last.name}")
                        last = None
                    else:
                        print("nothing to delete")
                    continue
                if cmd:
                    print(f"unknown command {cmd!r}")
                    continue
                last = await record_one(
                    args, robot, motion, follow, seen, session_dir, scenario, note, lag_s, rng, line
                )
                note = ""
                scenario = next_scenario(objects, recorded(args.raw), rng, args.held_out, factors)
        finally:
            follow.stop()
            await motion.close()
    n = len(list(session_dir.glob("ep_*/meta.json")))
    print(f"{n} episodes in {session_dir}")
    return 0


def describe(s: dict[str, Any], factors: dict[str, Any]) -> str:
    if s["negative"]:
        what = f"NEGATIVE: {s['object_id']} must stay OUT of view all episode"
    else:
        what = ", ".join(f"{k} {s[k]}" for k in factors)
    return f'{s["object_id"]} "{s["task_text"]}" | {what} | head {s["head_mode"]}'


async def record_one(
    args: argparse.Namespace,
    robot: Robot,
    motion: Motion,
    follow: Follow,
    seen: ObjectSource,
    session_dir: Path,
    scenario: dict[str, Any],
    note: str,
    lag_s: float,
    rng: random.Random,
    line: Any,
) -> Path | None:
    if scenario["head_mode"] == "baseline_jolt" and not await asyncio.to_thread(
        seen.begin, scenario["coco"]
    ):
        print("detector unavailable: recording with a sweep instead")
        scenario = {**scenario, "head_mode": "sweep"}
    if scenario["head_mode"] == "baseline_jolt":
        error, _ = await follow.start()
    else:
        error = motion.submit(sweep_move(scenario["range"] == "around", rng))
    if error:
        print(f"head busy: {error}")
        return None
    ep = EpisodeWriter(next_episode_dir(session_dir), open_video)
    stop = threading.Event()
    t0 = time.monotonic()
    threads = record_threads(robot, motion, ep, stop)
    print(f"recording {ep.path.name} for {args.seconds:.0f} s (Enter = stop early)...")
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(line(), args.seconds)
    stop.set()
    for th in threads:
        th.join(2.0)
    duration = time.monotonic() - t0
    if scenario["head_mode"] == "baseline_jolt":
        follow.stop()
    else:
        motion.stop()
    meta = {
        **{k: v for k, v in scenario.items() if k != "coco"},
        "target": args.target,
        "coco_label": scenario["coco"],
        "lighting": args.lighting,
        "background": args.background,
        "session": session_dir.name,
        "calib_id": args.calib_id,
        "camera_lag_s": lag_s,
        "t0": t0,
        "duration_s": round(duration, 2),
        "notes": note,
    }
    ep.close(meta)
    fps = ep.n_frames / duration if duration else 0.0
    print(f"saved {ep.path.name}: {ep.n_frames} frames ({fps:.1f} fps)")
    if fps < 10:
        print("  warning: under 10 fps, check Wi-Fi; consider deleting it (d)")
    await motion.settled()
    return ep.path


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", choices=("face", "object"), default="face")
    p.add_argument(
        "--targets",
        type=Path,
        required=True,
        help="people or prop list JSON (scripts/track_people.example.json, "
        "scripts/track_objects.example.json)",
    )
    p.add_argument("--lighting", choices=LIGHTING, required=True)
    p.add_argument(
        "--background",
        required=True,
        help=f"faces: {', '.join(FACE_BACKGROUND)}; objects: {', '.join(BACKGROUND)}",
    )
    p.add_argument("--seconds", type=float, default=25.0, help="episode length (default 25)")
    p.add_argument("--session", help="session folder name (default session_<today>_<letter>)")
    p.add_argument("--calib-id", help="calibration in use, e.g. calib_2026-11-20")
    p.add_argument(
        "--held-out", action="store_true", help="evaluation session: held-out targets only"
    )
    p.add_argument(
        "--raw", type=Path, help=f"raw data root (default {FACE_RAW_DIR}, objects {RAW_DIR})"
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main(parse())))
