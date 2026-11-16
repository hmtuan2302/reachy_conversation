"""CLI entry point. Subcommands are added per milestone (docs/architecture/06-operations.md §3)."""

from __future__ import annotations

import argparse
import logging
import sys

from reachy_demo import __version__
from reachy_demo.config import Config, ConfigError
from reachy_demo.ops.logging import configure_logging
from reachy_demo.voice import transcribe_cli

log = logging.getLogger("reachy_demo")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reachy-demo")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True, help="per milestone")
    sub.add_parser("check-config", help="validate .env / environment")
    transcribe_cli.add_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "transcribe":
        return transcribe_cli.run(args)  # needs only HF_TOKEN, not the robot config

    try:
        cfg = Config.from_env()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    configure_logging(cfg.log_dir, cfg.log_level, secrets=[cfg.openai_api_key])

    if args.command == "check-config":
        log.info("config ok", extra={"robot_host": cfg.robot_host})
    return 0


if __name__ == "__main__":
    sys.exit(main())
