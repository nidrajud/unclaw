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


def test_main_help_alias_prints_parser_help(capsys) -> None:
    assert unclaw_main.main(["help"]) == 0
    assert capsys.readouterr().out == unclaw_main.build_parser().format_help()


def test_main_dispatches_logs_with_default_simple(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Path | str | None] = {}

    def fake_logs(*, project_root: Path | None = None, mode: str = "simple") -> int:
        captured["project_root"] = project_root
        captured["mode"] = mode
        return 55

    monkeypatch.setattr(unclaw_main.logs_cli, "main", fake_logs)

    assert unclaw_main.main(["--project-root", str(tmp_path), "logs"]) == 55
    assert captured["project_root"] == tmp_path
    assert captured["mode"] == "simple"


def test_main_dispatches_logs_with_explicit_full(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Path | str | None] = {}

    def fake_logs(*, project_root: Path | None = None, mode: str = "simple") -> int:
        captured["project_root"] = project_root
        captured["mode"] = mode
        return 66

    monkeypatch.setattr(unclaw_main.logs_cli, "main", fake_logs)

    assert unclaw_main.main(["--project-root", str(tmp_path), "logs", "full"]) == 66
    assert captured["project_root"] == tmp_path
    assert captured["mode"] == "full"


def test_main_dispatches_update(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Path | None] = {}

    def fake_update(*, project_root: Path | None = None) -> int:
        captured["project_root"] = project_root
        return 44

    monkeypatch.setattr(unclaw_main, "update_main", fake_update)

    assert unclaw_main.main(["--project-root", str(tmp_path), "update"]) == 44
    assert captured["project_root"] == tmp_path
