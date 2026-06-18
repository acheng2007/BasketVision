"""Local chart generation for analysis timelines."""

from __future__ import annotations

from pathlib import Path


CHART_SERIES = {
    "shooting_elbow_angle": "Shooting Elbow Angle",
    "shooting_shoulder_angle": "Shooting Shoulder Angle",
    "shooting_hip_angle": "Shooting Hip Angle",
    "shooting_knee_angle": "Shooting Knee Angle",
    "guide_elbow_angle": "Guide Elbow Angle",
    "release_height_proxy": "Release Height Proxy",
}


def write_metric_charts(metrics: dict[str, object], output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    timeline = metrics.get("timeline", [])
    if not isinstance(timeline, list) or not timeline:
        return []

    chart_paths = []
    times = [float(frame["timestamp_ms"]) / 1000 for frame in timeline]

    for key, title in CHART_SERIES.items():
        values = [frame.get(f"{key}_smoothed", frame.get(key)) for frame in timeline]
        if not any(isinstance(value, int | float) for value in values):
            continue

        plt.figure(figsize=(10, 4))
        plt.plot(times, values, linewidth=2)
        plt.title(title)
        plt.xlabel("Time (seconds)")
        ylabel = "Normalized height" if key == "release_height_proxy" else "Degrees"
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        chart_path = output_dir / f"{key}.png"
        plt.savefig(chart_path)
        plt.close()
        chart_paths.append(str(chart_path))

    return chart_paths

