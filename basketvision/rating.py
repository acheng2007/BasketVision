"""Heuristic jump-shot form rating from pose metrics."""

from __future__ import annotations

import math
from typing import Any

# Ideal coaching ranges used for the first-pass scorer.
_TARGETS = {
    "set_elbow": (85.0, 105.0),  # shoulder-elbow-wrist near set point
    "release_elbow": (145.0, 175.0),  # more extended at release
    "release_shoulder": (70.0, 120.0),  # arm elevated
    "gather_knee": (95.0, 140.0),  # loaded legs
    "guide_elbow": (60.0, 140.0),  # soft guide hand, not locked out
    "release_height": (0.55, 0.95),  # wrist high in frame
}


def rate_shot(
    metrics: dict[str, Any],
    phases: dict[str, Any],
) -> dict[str, Any]:
    """Return an overall score (0-100) plus category breakdown and tips."""
    timeline = metrics.get("timeline") or []
    keyframes = phases.get("keyframes") or {}
    if not timeline:
        return {
            "overall": 0.0,
            "grade": "Incomplete",
            "categories": {},
            "tips": ["No pose data available to rate."],
            "snapshot": {},
        }

    snapshot = _phase_snapshot(timeline, keyframes)
    categories = {
        "set_point": _score_in_range(
            snapshot.get("set_elbow"),
            *_TARGETS["set_elbow"],
            soft_margin=25.0,
        ),
        "elbow_extension": _score_in_range(
            snapshot.get("release_elbow"),
            *_TARGETS["release_elbow"],
            soft_margin=30.0,
        ),
        "arm_elevation": _score_in_range(
            snapshot.get("release_shoulder"),
            *_TARGETS["release_shoulder"],
            soft_margin=35.0,
        ),
        "leg_load": _score_in_range(
            snapshot.get("gather_knee"),
            *_TARGETS["gather_knee"],
            soft_margin=35.0,
        ),
        "guide_hand": _score_in_range(
            snapshot.get("guide_elbow"),
            *_TARGETS["guide_elbow"],
            soft_margin=40.0,
        ),
        "release_height": _score_in_range(
            snapshot.get("release_height"),
            *_TARGETS["release_height"],
            soft_margin=0.25,
        ),
    }

    usable = [score for score in categories.values() if score is not None]
    overall = sum(usable) / len(usable) if usable else 0.0
    tips = _coaching_tips(snapshot, categories)

    return {
        "overall": round(overall, 1),
        "grade": _grade(overall),
        "categories": {
            name: None if score is None else round(score, 1) for name, score in categories.items()
        },
        "tips": tips,
        "snapshot": {key: _round_or_none(value) for key, value in snapshot.items()},
    }


def _phase_snapshot(
    timeline: list[dict[str, Any]],
    keyframes: dict[str, Any],
) -> dict[str, float | None]:
    by_frame = {int(frame["frame_index"]): frame for frame in timeline}

    def frame_for(phase: str) -> dict[str, Any] | None:
        ref = keyframes.get(phase)
        if not isinstance(ref, dict):
            return None
        return by_frame.get(int(ref["frame_index"]))

    gather = frame_for("gather") or timeline[0]
    set_point = frame_for("set_point") or timeline[len(timeline) // 3]
    release = frame_for("release") or timeline[len(timeline) // 2]

    return {
        "set_elbow": _metric(set_point, "shooting_elbow_angle"),
        "release_elbow": _metric(release, "shooting_elbow_angle"),
        "release_shoulder": _metric(release, "shooting_shoulder_angle"),
        "gather_knee": _metric(gather, "shooting_knee_angle"),
        "guide_elbow": _metric(release, "guide_elbow_angle"),
        "release_height": _metric(release, "release_height_proxy"),
    }


def _metric(frame: dict[str, Any] | None, key: str) -> float | None:
    if not frame:
        return None
    smoothed = frame.get(f"{key}_smoothed")
    if _is_number(smoothed):
        return float(smoothed)
    value = frame.get(key)
    if _is_number(value):
        return float(value)
    return None


def _score_in_range(
    value: float | None,
    low: float,
    high: float,
    *,
    soft_margin: float,
) -> float | None:
    if value is None or not _is_number(value):
        return None
    if low <= value <= high:
        return 100.0
    distance = low - value if value < low else value - high
    if distance >= soft_margin:
        return 0.0
    return max(0.0, 100.0 * (1.0 - distance / soft_margin))


def _coaching_tips(
    snapshot: dict[str, float | None],
    categories: dict[str, float | None],
) -> list[str]:
    tips: list[str] = []

    set_elbow = snapshot.get("set_elbow")
    set_score = categories.get("set_point")
    if set_score is not None and set_score < 70 and set_elbow is not None:
        if set_elbow < _TARGETS["set_elbow"][0]:
            tips.append("Open the shooting elbow a bit more at the set point.")
        else:
            tips.append("Keep the shooting elbow closer to a right angle at the set point.")

    release_elbow = snapshot.get("release_elbow")
    elbow_score = categories.get("elbow_extension")
    if (
        elbow_score is not None
        and elbow_score < 70
        and release_elbow is not None
        and release_elbow < _TARGETS["release_elbow"][0]
    ):
        tips.append("Extend through the shot - finish with a higher elbow lockout.")

    gather_knee = snapshot.get("gather_knee")
    leg_score = categories.get("leg_load")
    if leg_score is not None and leg_score < 70 and gather_knee is not None:
        if gather_knee > _TARGETS["gather_knee"][1]:
            tips.append("Bend the knees more on the gather to load power into the shot.")
        else:
            tips.append("Avoid over-squatting - keep the gather athletic and balanced.")

    release_height = snapshot.get("release_height")
    height_score = categories.get("release_height")
    if (
        height_score is not None
        and height_score < 70
        and release_height is not None
        and release_height < _TARGETS["release_height"][0]
    ):
        tips.append("Release higher - get the ball above the forehead before the flick.")

    guide = snapshot.get("guide_elbow")
    guide_score = categories.get("guide_hand")
    if guide_score is not None and guide_score < 70 and guide is not None:
        tips.append("Keep the guide hand soft on the side of the ball.")

    if not tips:
        tips.append("Solid look - keep repeating this motion for consistency.")

    return tips[:3]


def _grade(score: float) -> str:
    if score >= 90:
        return "Elite"
    if score >= 80:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Fair"
    if score >= 40:
        return "Needs work"
    return "Rough"


def _round_or_none(value: float | None) -> float | None:
    if value is None or not _is_number(value):
        return None
    return round(float(value), 2)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not math.isnan(float(value))
