"""Unified command entrypoint for the Unclaw runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from unclaw import __version__
from unclaw.channels import cli as cli_channel
from unclaw.channels import telegram_bot
from unclaw.onboarding import main as onboarding_main
from unclaw.update import main as update_main


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the user-facing `unclaw` command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    command_name = args.command or "start"
    project_root = args.project_root

    if command_name == "start":
        return cli_channel.main(project_root=project_root)
    if command_name == "telegram":
        return telegram_bot.main(project_root=project_root)
    if command_name == "onboard":
        return onboarding_main(project_root=project_root)
    if command_name == "update":
        return update_main(project_root=project_root)

    parser.error(f"Unsupported command: {command_name}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    """Build the small public command parser."""

    parser = argparse.ArgumentParser(
        prog="unclaw",
        description="Local-first AI agent runtime.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("start", help="Start the terminal runtime.")
    subparsers.add_parser("telegram", help="Start the Telegram channel.")
    subparsers.add_parser("onboard", help="Run interactive onboarding.")
    subparsers.add_parser("update", help="Fetch and fast-forward the local checkout.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
