"""Tests for user-facing command-line behavior."""

from __future__ import annotations

import argparse
import sys

import pytest

from basketvision import cli


def test_positive_int_rejects_zero() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="at least 1"):
        cli.positive_int("0")


def test_live_entry_point_preserves_user_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["basketvision-live", "--camera", "2", "--no-mirror"])
    monkeypatch.setattr(cli, "main", lambda argv=None: received.append(argv))

    cli.main_live()

    assert received == [["live", "--camera", "2", "--no-mirror"]]


def test_missing_video_reports_parser_error_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["/tmp/basketvision-missing-video.mp4"])

    assert raised.value.code == 2
    assert "Video not found" in capsys.readouterr().err
