"""Tests for analysis output lifecycle."""

from pathlib import Path

from basketvision.analysis import _remove_stale_visuals


def test_remove_stale_visuals_only_removes_generated_file_types(tmp_path: Path) -> None:
    keyframes = tmp_path / "keyframes"
    charts = tmp_path / "charts"
    keyframes.mkdir()
    charts.mkdir()
    stale_keyframe = keyframes / "release.jpg"
    stale_chart = charts / "elbow.png"
    preserved_file = charts / "notes.txt"
    stale_keyframe.touch()
    stale_chart.touch()
    preserved_file.touch()

    _remove_stale_visuals(keyframes, charts)

    assert not stale_keyframe.exists()
    assert not stale_chart.exists()
    assert preserved_file.exists()
