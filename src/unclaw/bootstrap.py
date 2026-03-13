"""Startup preparation logic."""

from __future__ import annotations

from pathlib import Path

from unclaw.errors import BootstrapError
from unclaw.settings import Settings, load_settings


def bootstrap(project_root: Path | None = None) -> Settings:
    settings = load_settings(project_root=project_root)
    prepare_runtime(settings)
    return settings


def prepare_runtime(settings: Settings) -> None:
    try:
        for directory in settings.paths.runtime_directories():
            directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BootstrapError("Could not prepare the Unclaw runtime directories.") from exc
