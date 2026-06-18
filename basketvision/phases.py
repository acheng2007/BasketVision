"""Heuristic shot phase detection."""

from __future__ import annotations


PHASES = ["gather", "set_point", "release", "follow_through", "landing"]


def detect_phases(metrics: dict[str, object]) -> dict[str, object]:
    timeline = metrics.get("timeline", [])
    if not isinstance(timeline, list) or not timeline:
        return {"phases": [], "keyframes": {}}

    release_idx = _index_of_max(timeline, "release_height_proxy_smoothed")
    if release_idx is None:
        release_idx = _index_of_max(timeline, "release_height_proxy")
    if release_idx is None:
        release_idx = len(timeline) // 2

    set_point_idx = max(0, int(release_idx * 0.72))
    follow_idx = min(len(timeline) - 1, release_idx + max(1, int(len(timeline) * 0.08)))
    landing_idx = len(timeline) - 1

    gather_end = max(0, set_point_idx - 1)
    set_end = max(set_point_idx, release_idx - 1)
    follow_end = max(follow_idx, min(len(timeline) - 1, release_idx + int(len(timeline) * 0.2)))

    labeled = []
    for idx, frame in enumerate(timeline):
        if idx <= gather_end:
            phase = "gather"
        elif idx <= set_end:
            phase = "set_point"
        elif idx == release_idx:
            phase = "release"
        elif idx <= follow_end:
            phase = "follow_through"
        else:
            phase = "landing"

        labeled.append(
            {
                "frame_index": frame["frame_index"],
                "timestamp_ms": frame["timestamp_ms"],
                "phase": phase,
            }
        )

    keyframes = {
        "gather": _frame_ref(timeline[0]),
        "set_point": _frame_ref(timeline[set_point_idx]),
        "release": _frame_ref(timeline[release_idx]),
        "follow_through": _frame_ref(timeline[follow_idx]),
        "landing": _frame_ref(timeline[landing_idx]),
    }

    return {
        "phases": labeled,
        "keyframes": keyframes,
        "method": "heuristic_release_height",
    }


def phase_lookup(phases: dict[str, object]) -> dict[int, str]:
    rows = phases.get("phases", [])
    if not isinstance(rows, list):
        return {}
    return {int(row["frame_index"]): str(row["phase"]) for row in rows}


def _index_of_max(timeline: list[dict[str, object]], key: str) -> int | None:
    best_idx = None
    best_value = None
    for idx, frame in enumerate(timeline):
        value = frame.get(key)
        if not isinstance(value, int | float):
            continue
        if best_value is None or float(value) > best_value:
            best_idx = idx
            best_value = float(value)
    return best_idx


def _frame_ref(frame: dict[str, object]) -> dict[str, int]:
    return {
        "frame_index": int(frame["frame_index"]),
        "timestamp_ms": int(frame["timestamp_ms"]),
    }

