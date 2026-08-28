"""Tests for jump-shot form rating and live detection heuristics."""

from __future__ import annotations

from basketvision.live_detector import ShotDetector
from basketvision.rating import rate_shot


def _frame(
    frame_index: int,
    timestamp_ms: int,
    *,
    elbow: float,
    shoulder: float,
    knee: float,
    guide: float,
    height: float,
) -> dict[str, float | int]:
    return {
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "shooting_elbow_angle": elbow,
        "shooting_shoulder_angle": shoulder,
        "shooting_hip_angle": 150.0,
        "shooting_knee_angle": knee,
        "guide_elbow_angle": guide,
        "release_height_proxy": height,
    }


def test_rate_shot_scores_strong_form_high() -> None:
    timeline = [
        _frame(0, 0, elbow=100, shoulder=40, knee=115, guide=90, height=0.35),
        _frame(1, 40, elbow=95, shoulder=55, knee=120, guide=95, height=0.45),
        _frame(2, 80, elbow=95, shoulder=90, knee=130, guide=100, height=0.65),
        _frame(3, 120, elbow=160, shoulder=95, knee=155, guide=110, height=0.78),
        _frame(4, 160, elbow=165, shoulder=100, knee=165, guide=115, height=0.72),
        _frame(5, 200, elbow=150, shoulder=85, knee=170, guide=100, height=0.55),
    ]
    metrics = {"timeline": timeline, "summary": {}}
    phases = {
        "keyframes": {
            "gather": {"frame_index": 0, "timestamp_ms": 0},
            "set_point": {"frame_index": 2, "timestamp_ms": 80},
            "release": {"frame_index": 3, "timestamp_ms": 120},
            "follow_through": {"frame_index": 4, "timestamp_ms": 160},
            "landing": {"frame_index": 5, "timestamp_ms": 200},
        }
    }

    rating = rate_shot(metrics, phases)
    assert rating["overall"] >= 80
    assert rating["grade"] in {"Good", "Excellent", "Elite"}
    assert rating["tips"]


def test_rate_shot_flags_stiff_legs() -> None:
    timeline = [
        _frame(0, 0, elbow=95, shoulder=40, knee=175, guide=90, height=0.35),
        _frame(1, 40, elbow=95, shoulder=90, knee=175, guide=95, height=0.60),
        _frame(2, 80, elbow=160, shoulder=95, knee=175, guide=100, height=0.75),
    ]
    metrics = {"timeline": timeline, "summary": {}}
    phases = {
        "keyframes": {
            "gather": {"frame_index": 0, "timestamp_ms": 0},
            "set_point": {"frame_index": 1, "timestamp_ms": 40},
            "release": {"frame_index": 2, "timestamp_ms": 80},
        }
    }

    rating = rate_shot(metrics, phases)
    assert rating["categories"]["leg_load"] is not None
    assert rating["categories"]["leg_load"] < 70
    assert any("knee" in tip.lower() or "bend" in tip.lower() for tip in rating["tips"])


def _synthetic_landmarks(
    *,
    wrist_y: float,
    knee_angle_proxy: float = 0.55,
) -> list[dict[str, float]]:
    """Build a minimal 33-landmark pose with controllable right-wrist height."""
    landmarks = [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0, "presence": 1.0} for _ in range(33)
    ]
    # Torso
    landmarks[11].update({"x": 0.45, "y": 0.40})  # left shoulder
    landmarks[12].update({"x": 0.55, "y": 0.40})  # right shoulder
    landmarks[13].update({"x": 0.40, "y": 0.55})  # left elbow
    landmarks[14].update({"x": 0.60, "y": 0.55})  # right elbow
    landmarks[15].update({"x": 0.38, "y": 0.70})  # left wrist
    landmarks[16].update({"x": 0.62, "y": wrist_y})  # right wrist
    landmarks[23].update({"x": 0.46, "y": 0.65})  # left hip
    landmarks[24].update({"x": 0.54, "y": 0.65})  # right hip
    landmarks[25].update({"x": 0.46, "y": knee_angle_proxy})  # left knee
    landmarks[26].update({"x": 0.54, "y": knee_angle_proxy})  # right knee
    landmarks[27].update({"x": 0.46, "y": 0.90})  # left ankle
    landmarks[28].update({"x": 0.54, "y": 0.90})  # right ankle
    return landmarks


def test_shot_detector_emits_rating_after_rise_and_drop() -> None:
    detector = ShotDetector(
        shooting_hand="right",
        pre_roll_ms=200,
        post_roll_ms=120,
        min_shot_ms=200,
        rise_threshold=0.06,
        peak_min_height=0.50,
        cooldown_ms=100,
    )

    # Idle / baseline near the waist.
    for idx in range(10):
        result = detector.update(
            frame_index=idx,
            timestamp_ms=idx * 40,
            landmarks=_synthetic_landmarks(wrist_y=0.70),
        )
        assert result is None

    # Rising shot.
    heights = [0.62, 0.55, 0.45, 0.35, 0.28, 0.25]
    detected = None
    frame_index = 10
    for wrist_y in heights:
        detected = detector.update(
            frame_index=frame_index,
            timestamp_ms=frame_index * 40,
            landmarks=_synthetic_landmarks(wrist_y=wrist_y),
        )
        frame_index += 1
        if detected is not None:
            break

    # Hold near peak, then drop for follow-through / landing.
    if detected is None:
        for wrist_y in [0.25, 0.27, 0.30, 0.40, 0.55, 0.65, 0.70, 0.72]:
            detected = detector.update(
                frame_index=frame_index,
                timestamp_ms=frame_index * 40,
                landmarks=_synthetic_landmarks(wrist_y=wrist_y),
            )
            frame_index += 1
            if detected is not None:
                break

    assert detected is not None
    assert detected.shot_id == 1
    assert "overall" in detected.rating
    assert detected.rating["overall"] >= 0


def test_shot_detector_starts_on_gradual_natural_wrist_rise() -> None:
    detector = ShotDetector(shooting_hand="right")
    frame_index = 0
    for _ in range(12):
        detector.update(
            frame_index=frame_index,
            timestamp_ms=frame_index * 40,
            landmarks=_synthetic_landmarks(wrist_y=0.72),
        )
        frame_index += 1

    # A smooth gather should still cross the baseline instead of being absorbed.
    for wrist_y in [0.70, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56]:
        detector.update(
            frame_index=frame_index,
            timestamp_ms=frame_index * 40,
            landmarks=_synthetic_landmarks(wrist_y=wrist_y),
        )
        frame_index += 1

    assert detector.status == "shooting"


def test_shot_detector_uses_either_visible_wrist_for_start() -> None:
    detector = ShotDetector(shooting_hand="right")
    for frame_index in range(10):
        detector.update(
            frame_index=frame_index,
            timestamp_ms=frame_index * 40,
            landmarks=_synthetic_landmarks(wrist_y=0.72),
        )

    landmarks = _synthetic_landmarks(wrist_y=0.72)
    landmarks[15]["y"] = 0.50
    detector.update(frame_index=10, timestamp_ms=400, landmarks=landmarks)
    assert detector.status == "shooting"
