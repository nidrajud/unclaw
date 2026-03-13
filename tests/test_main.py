from __future__ import annotations

from pathlib import Path

from unclaw import main as unclaw_main


def test_main_defaults_to_start(monkeypatch) -> None:
    captured: dict[str, Path | None] = {}

    def fake_start(*, project_root: Path | None = None) -> int:
        captured["project_root"] = project_root
        return 11

    monkeypatch.setattr(unclaw_main.cli_channel, "main", fake_start)

    assert unclaw_main.main([]) == 11
    assert captured["project_root"] is None


def test_main_dispatches_telegram_with_project_root(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Path | None] = {}

    def fake_telegram(*, project_root: Path | None = None) -> int:
        captured["project_root"] = project_root
        return 22

    monkeypatch.setattr(unclaw_main.telegram_bot, "main", fake_telegram)

    assert unclaw_main.main(["--project-root", str(tmp_path), "telegram"]) == 22
    assert captured["project_root"] == tmp_path


def test_main_dispatches_onboarding(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Path | None] = {}

    def fake_onboarding(*, project_root: Path | None = None) -> int:
        captured["project_root"] = project_root
        return 33

    monkeypatch.setattr(unclaw_main, "onboarding_main", fake_onboarding)

    assert unclaw_main.main(["--project-root", str(tmp_path), "onboard"]) == 33
    assert captured["project_root"] == tmp_path
