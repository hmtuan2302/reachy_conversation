"""CLI entry point. Subcommands are added per milestone (docs/architecture/06-operations.md §3)."""

from __future__ import annotations

import argparse
import logging
import sys

from reachy_demo import __version__
from reachy_demo.config import Config, ConfigError
from reachy_demo.ops.logging import configure_logging

log = logging.getLogger("reachy_demo")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reachy-demo")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "command", choices=["check-config"], help="more commands arrive per milestone"
    )
    args = parser.parse_args(argv)

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
