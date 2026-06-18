"""Markdown report generation."""

from __future__ import annotations

import math
from pathlib import Path


METRIC_LABELS = {
    "shooting_elbow_angle": "Shooting elbow",
    "shooting_shoulder_angle": "Shooting shoulder",
    "shooting_hip_angle": "Hip load",
    "shooting_knee_angle": "Knee bend",
    "guide_elbow_angle": "Guide elbow",
    "release_height_proxy": "Release height proxy",
}


def write_report(
    output_path: Path,
    *,
    video_path: Path,
    metadata: dict[str, object],
    metrics: dict[str, object],
    phases: dict[str, object],
    chart_paths: list[str],
    ball_tracking_enabled: bool = False,
) -> None:
    summary = metrics.get("summary", {})
    keyframes = phases.get("keyframes", {})

    lines = [
        "# BasketVision Analysis Report",
        "",
        f"- **Video:** `{video_path}`",
        f"- **Duration:** {float(metadata.get('duration_seconds', 0.0)):.2f}s",
        f"- **Resolution:** {metadata.get('width')}x{metadata.get('height')}",
        f"- **FPS:** {float(metadata.get('fps', 0.0)):.2f}",
        f"- **Detected pose frames:** {metrics.get('detected_pose_frames', 0)}",
        "",
        "## Shot Phase Key Frames",
        "",
    ]

    if isinstance(keyframes, dict):
        for phase, ref in keyframes.items():
            if isinstance(ref, dict):
                lines.append(
                    f"- **{phase.replace('_', ' ').title()}:** "
                    f"frame {ref.get('frame_index')} at {ref.get('timestamp_ms')}ms"
                )

    lines.extend(["", "## Metric Summary", ""])

    if isinstance(summary, dict):
        lines.append("| Metric | Min | Max | Mean | Range |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for key, label in METRIC_LABELS.items():
            item = summary.get(key)
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {label} | {_fmt(item.get('min'))} | {_fmt(item.get('max'))} | "
                f"{_fmt(item.get('mean'))} | {_fmt(item.get('range'))} |"
            )

    lines.extend(
        [
            "",
            "## Coaching Notes",
            "",
            "- This is a first-pass pose-only analysis, so treat the feedback as directional.",
            "- Use clear side-view or 45-degree footage for better joint-angle estimates.",
            "- The release frame is currently estimated from highest shooting-wrist position.",
            _ball_tracking_note(ball_tracking_enabled),
            "",
            "## Generated Visuals",
            "",
            "- `annotated.mp4` contains the skeleton overlay and rough phase labels.",
            "- `keyframes/` contains still images for the detected shot phases.",
            "- `charts/` contains local timeline charts.",
        ]
    )

    for path in chart_paths:
        lines.append(f"- `{Path(path).name}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ball_tracking_note(enabled: bool) -> str:
    if enabled:
        return "- Ball/hoop detections were saved to `ball_tracking.json`."
    return "- Ball and hoop tracking are optional and disabled by default until pose metrics are reliable."


def _fmt(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    if math.isnan(float(value)):
        return "n/a"
    return f"{float(value):.2f}"

