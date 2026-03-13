"""Main entry point for the Unclaw runtime."""

from __future__ import annotations

import sys

from unclaw.bootstrap import bootstrap
from unclaw.errors import UnclawError
from unclaw.settings import Settings


def main() -> int:
    try:
        settings = bootstrap()
    except UnclawError as exc:
        print(f"Failed to start Unclaw: {exc}", file=sys.stderr)
        return 1

    _print_startup_summary(settings)
    return 0


def _print_startup_summary(settings: Settings) -> None:
    print(f"{settings.app.display_name} 🦐 ready")
    print(
        "Environment: "
        f"{settings.app.environment} | "
        f"Default model: {settings.app.default_model_profile} "
        f"({settings.default_model.model_name})"
    )
    print("Runtime paths:")
    for label, path in _runtime_path_rows(settings):
        print(f"  {label:<14} {path}")


def _runtime_path_rows(settings: Settings) -> tuple[tuple[str, str], ...]:
    return (
        ("project_root", str(settings.paths.project_root)),
        ("config_dir", str(settings.paths.config_dir)),
        ("data_dir", str(settings.paths.data_dir)),
        ("database", str(settings.paths.database_path)),
        ("logs_dir", str(settings.paths.logs_dir)),
        ("log_file", str(settings.paths.log_file_path)),
        ("sessions_dir", str(settings.paths.sessions_dir)),
        ("cache_dir", str(settings.paths.cache_dir)),
        ("files_dir", str(settings.paths.files_dir)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
