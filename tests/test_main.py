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
    captured: dict[str, Path | str | int | None] = {}

    def fake_telegram(
        *,
        project_root: Path | None = None,
        command: str = "start",
        chat_id: int | None = None,
    ) -> int:
        captured["project_root"] = project_root
        captured["command"] = command
        captured["chat_id"] = chat_id
        return 22

    monkeypatch.setattr(unclaw_main.telegram_bot, "main", fake_telegram)

    assert unclaw_main.main(["--project-root", str(tmp_path), "telegram"]) == 22
    assert captured["project_root"] == tmp_path
    assert captured["command"] == "start"
    assert captured["chat_id"] is None


def test_main_dispatches_telegram_management_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Path | str | int | None] = {}

    def fake_telegram(
        *,
        project_root: Path | None = None,
        command: str = "start",
        chat_id: int | None = None,
    ) -> int:
        captured["project_root"] = project_root
        captured["command"] = command
        captured["chat_id"] = chat_id
        return 23

    monkeypatch.setattr(unclaw_main.telegram_bot, "main", fake_telegram)

    assert (
        unclaw_main.main(
            ["--project-root", str(tmp_path), "telegram", "allow", "123456789"]
        )
        == 23
    )
    assert captured["project_root"] == tmp_path
    assert captured["command"] == "allow"
    assert captured["chat_id"] == 123456789


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


def test_main_help_mentions_telegram_management_commands(capsys) -> None:
    unclaw_main.main(["help"])
    help_text = capsys.readouterr().out

    assert "unclaw telegram allow-latest" in help_text
    assert "unclaw telegram list" in help_text


def test_main_help_uses_logs_as_the_canonical_simple_command(capsys) -> None:
    unclaw_main.main(["help"])
    help_text = capsys.readouterr().out

    assert "  unclaw logs\n" in help_text
    assert "  unclaw logs full\n" in help_text
    assert "  unclaw logs simple\n" not in help_text


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


def test_main_dispatches_logs_with_backward_compatible_simple_alias(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Path | str | None] = {}

    def fake_logs(*, project_root: Path | None = None, mode: str = "simple") -> int:
        captured["project_root"] = project_root
        captured["mode"] = mode
        return 77

    monkeypatch.setattr(unclaw_main.logs_cli, "main", fake_logs)

    assert unclaw_main.main(["--project-root", str(tmp_path), "logs", "simple"]) == 77
    assert captured["project_root"] == tmp_path
    assert captured["mode"] == "simple"


def test_main_dispatches_update(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Path | None] = {}

    def fake_update(*, project_root: Path | None = None) -> int:
        captured["project_root"] = project_root
        return 44

    monkeypatch.setattr(unclaw_main, "update_main", fake_update)

    assert unclaw_main.main(["--project-root", str(tmp_path), "update"]) == 44
    assert captured["project_root"] == tmp_path
