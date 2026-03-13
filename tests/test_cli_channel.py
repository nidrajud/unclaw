from __future__ import annotations

from unclaw.channels.cli import _TerminalAssistantStream


def test_terminal_assistant_stream_finishes_cleanly_when_stream_matches(capsys) -> None:
    stream = _TerminalAssistantStream()

    stream.write("Hel")
    stream.write("lo")
    stream.finish("Hello")

    assert capsys.readouterr().out == "Unclaw> Hello\n"


def test_terminal_assistant_stream_explains_interrupted_streams(capsys) -> None:
    stream = _TerminalAssistantStream()

    stream.write("Partial")
    stream.finish("Saved reply")

    assert capsys.readouterr().out == (
        "Unclaw> Partial\n"
        "[stream interrupted; showing the saved final reply]\n"
        "Unclaw> Saved reply\n"
    )
