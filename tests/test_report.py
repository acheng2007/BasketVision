"""Tests for generated analysis reports."""

from pathlib import Path

from basketvision.report import write_report


def test_report_explains_when_no_pose_was_detected(tmp_path: Path) -> None:
    output_path = tmp_path / "report.md"

    write_report(
        output_path,
        video_path=Path("shot.mp4"),
        metadata={"duration_seconds": 1.0, "width": 640, "height": 360, "fps": 30.0},
        metrics={"detected_pose_frames": 0, "summary": {}},
        phases={"keyframes": {}},
        chart_paths=[],
    )

    report = output_path.read_text(encoding="utf-8")
    assert "No shot phases were detected" in report
    assert "No pose metrics were available" in report
    assert "No keyframe images were generated" in report
    assert "No metric charts were generated" in report
    assert "`keyframes/` contains" not in report
