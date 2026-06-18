"""Shot-form metrics derived from pose landmarks."""

from __future__ import annotations

import math

from basketvision.geometry import angle_degrees, landmark_is_usable, midpoint
from basketvision.landmarks import LANDMARK_IDS, SIDE_LANDMARKS


ANGLE_KEYS = [
    "shooting_elbow_angle",
    "shooting_shoulder_angle",
    "shooting_hip_angle",
    "shooting_knee_angle",
    "guide_elbow_angle",
    "release_height_proxy",
]


def compute_metrics(
    landmark_frames: list[dict[str, object]],
    *,
    shooting_hand: str = "right",
) -> dict[str, object]:
    timeline = []

    for frame in landmark_frames:
        landmarks = frame.get("landmarks") or []
        if not landmarks:
            continue

        frame_metrics = compute_frame_metrics(
            landmarks, shooting_hand=shooting_hand, timestamp_ms=int(frame["timestamp_ms"])
        )
        frame_metrics["frame_index"] = int(frame["frame_index"])
        timeline.append(frame_metrics)

    smoothed = _smooth_timeline(timeline)

    return {
        "shooting_hand": shooting_hand,
        "frame_count": len(landmark_frames),
        "detected_pose_frames": len(timeline),
        "timeline": smoothed,
        "summary": summarize_timeline(smoothed),
    }


def compute_frame_metrics(
    landmarks: list[dict[str, float]],
    *,
    shooting_hand: str,
    timestamp_ms: int,
) -> dict[str, float | int]:
    guide_hand = "left" if shooting_hand == "right" else "right"
    shooting = SIDE_LANDMARKS[shooting_hand]
    guide = SIDE_LANDMARKS[guide_hand]

    metrics: dict[str, float | int] = {"timestamp_ms": timestamp_ms}

    metrics["shooting_elbow_angle"] = _safe_angle(
        landmarks, shooting["shoulder"], shooting["elbow"], shooting["wrist"]
    )
    metrics["shooting_shoulder_angle"] = _safe_angle(
        landmarks, shooting["hip"], shooting["shoulder"], shooting["elbow"]
    )
    metrics["shooting_knee_angle"] = _safe_angle(
        landmarks, shooting["hip"], shooting["knee"], shooting["ankle"]
    )
    metrics["guide_elbow_angle"] = _safe_angle(
        landmarks, guide["shoulder"], guide["elbow"], guide["wrist"]
    )

    shoulder_mid = _safe_midpoint(
        landmarks, LANDMARK_IDS["left_shoulder"], LANDMARK_IDS["right_shoulder"]
    )
    hip_mid = _safe_midpoint(landmarks, LANDMARK_IDS["left_hip"], LANDMARK_IDS["right_hip"])
    if shoulder_mid and hip_mid and _usable(landmarks, shooting["knee"]):
        metrics["shooting_hip_angle"] = angle_degrees(
            shoulder_mid, hip_mid, landmarks[shooting["knee"]]
        )
    else:
        metrics["shooting_hip_angle"] = math.nan

    if _usable(landmarks, shooting["wrist"]):
        # MediaPipe y grows downward, so 1-y is a convenient normalized height proxy.
        metrics["release_height_proxy"] = 1.0 - float(landmarks[shooting["wrist"]]["y"])
    else:
        metrics["release_height_proxy"] = math.nan

    return metrics


def summarize_timeline(timeline: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    summary = {}
    for key in ANGLE_KEYS:
        values = [float(frame[key]) for frame in timeline if _is_number(frame.get(key))]
        if not values:
            summary[key] = {"min": math.nan, "max": math.nan, "mean": math.nan, "range": math.nan}
            continue
        minimum = min(values)
        maximum = max(values)
        summary[key] = {
            "min": minimum,
            "max": maximum,
            "mean": sum(values) / len(values),
            "range": maximum - minimum,
        }
    return summary


def _safe_angle(
    landmarks: list[dict[str, float]],
    a_idx: int,
    b_idx: int,
    c_idx: int,
) -> float:
    if not (_usable(landmarks, a_idx) and _usable(landmarks, b_idx) and _usable(landmarks, c_idx)):
        return math.nan
    return angle_degrees(landmarks[a_idx], landmarks[b_idx], landmarks[c_idx])


def _safe_midpoint(
    landmarks: list[dict[str, float]], left_idx: int, right_idx: int
) -> dict[str, float] | None:
    if not (_usable(landmarks, left_idx) and _usable(landmarks, right_idx)):
        return None
    return midpoint([landmarks[left_idx], landmarks[right_idx]])


def _usable(landmarks: list[dict[str, float]], idx: int) -> bool:
    return idx < len(landmarks) and landmark_is_usable(landmarks[idx])


def _smooth_timeline(timeline: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    if len(timeline) < 7:
        return timeline

    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return timeline

    smoothed = [dict(frame) for frame in timeline]
    window = min(len(timeline) if len(timeline) % 2 else len(timeline) - 1, 11)
    if window < 7:
        return timeline

    for key in ANGLE_KEYS:
        values = [float(frame[key]) if _is_number(frame.get(key)) else math.nan for frame in timeline]
        filled = _fill_missing(values)
        if filled is None:
            continue
        filtered = savgol_filter(filled, window_length=window, polyorder=2)
        for frame, value in zip(smoothed, filtered, strict=True):
            frame[f"{key}_smoothed"] = float(value)

    return smoothed


def _fill_missing(values: list[float]) -> list[float] | None:
    valid = [(idx, value) for idx, value in enumerate(values) if _is_number(value)]
    if len(valid) < 3:
        return None

    filled = values[:]
    last_valid = valid[0][1]
    for idx, value in enumerate(filled):
        if _is_number(value):
            last_valid = value
        else:
            filled[idx] = last_valid

    next_valid = valid[-1][1]
    for idx in range(len(filled) - 1, -1, -1):
        if _is_number(filled[idx]):
            next_valid = filled[idx]
        else:
            filled[idx] = next_valid

    return filled


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not math.isnan(float(value))

