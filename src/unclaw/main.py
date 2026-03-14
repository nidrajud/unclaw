"""Unified command entrypoint for the Unclaw runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from unclaw import __version__
from unclaw.channels import cli as cli_channel
from unclaw.channels import telegram_bot
from unclaw.logs import cli as logs_cli
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
        return telegram_bot.main(
            project_root=project_root,
            command=args.telegram_command or "start",
            chat_id=getattr(args, "chat_id", None),
        )
    if command_name == "onboard":
        return onboarding_main(project_root=project_root)
    if command_name == "help":
        parser.print_help()
        return 0
    if command_name == "logs":
        return logs_cli.main(project_root=project_root, mode=args.mode)
    if command_name == "update":
        return update_main(project_root=project_root)

    parser.error(f"Unsupported command: {command_name}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    """Build the small public command parser."""

    parser = argparse.ArgumentParser(
        prog="unclaw",
        description="Local-first AI runtime for your machine.",
        epilog=(
            "Examples:\n"
            "  unclaw start\n"
            "  unclaw help\n"
            "  unclaw logs\n"
            "  unclaw logs full\n"
            "  unclaw onboard\n"
            "  unclaw telegram\n"
            "  unclaw telegram allow-latest\n"
            "  unclaw telegram allow 123456789\n"
            "  unclaw telegram list\n"
            "  unclaw update"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    subparsers.add_parser("start", help="Start the local terminal chat.")
    telegram_parser = subparsers.add_parser(
        "telegram",
        help="Start the Telegram bot or manage the local chat allowlist.",
        description=(
            "Start the Telegram bot channel or manage the local Telegram chat "
            "allowlist."
        ),
        epilog=(
            "Examples:\n"
            "  unclaw telegram\n"
            "  unclaw telegram start\n"
            "  unclaw telegram list\n"
            "  unclaw telegram allow 123456789\n"
            "  unclaw telegram allow-latest\n"
            "  unclaw telegram revoke 123456789"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    telegram_parser.set_defaults(telegram_command="start")
    telegram_subparsers = telegram_parser.add_subparsers(dest="telegram_command")
    telegram_subparsers.add_parser(
        "start",
        help="Start the Telegram bot channel.",
    )
    telegram_subparsers.add_parser(
        "list",
        help="List authorized Telegram chats and the latest rejected chat.",
    )
    allow_parser = telegram_subparsers.add_parser(
        "allow",
        help="Authorize one Telegram chat id locally.",
    )
    allow_parser.add_argument("chat_id", type=int, help="Numeric Telegram chat id.")
    telegram_subparsers.add_parser(
        "allow-latest",
        help="Authorize the most recently rejected Telegram chat id.",
    )
    revoke_parser = telegram_subparsers.add_parser(
        "revoke",
        help="Remove one Telegram chat id from the local allowlist.",
    )
    revoke_parser.add_argument("chat_id", type=int, help="Numeric Telegram chat id.")
    subparsers.add_parser("onboard", help="Run guided local setup.")
    subparsers.add_parser("help", help="Show the same help as `unclaw --help`.")
    logs_parser = subparsers.add_parser(
        "logs",
        help="Read local runtime logs.",
        description=(
            "Show local runtime logs. `unclaw logs` uses the simple view by "
            "default, and `unclaw logs full` shows the raw JSON stream."
        ),
    )
    logs_parser.add_argument(
        "mode",
        nargs="?",
        choices=("simple", "full"),
        default="simple",
        metavar="mode",
        help=(
            "Optional log view. Leave empty for the simple view, or use "
            "`full` for raw JSON. `simple` remains accepted as a compatibility alias."
        ),
    )
    subparsers.add_parser(
        "update",
        help="Safely fetch and fast-forward this local checkout.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
