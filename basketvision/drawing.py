"""Frame annotation and live dashboard rendering helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from basketvision.landmarks import POSE_CONNECTIONS, SIDE_LANDMARKS

BG, CARD, CARD_ALT = (18, 16, 14), (31, 28, 25), (40, 36, 32)
TEXT, MUTED = (244, 244, 242), (160, 157, 151)
ACCENT, GREEN, ORANGE, RED = (60, 205, 255), (116, 218, 111), (74, 155, 255), (91, 91, 242)
RECORD_BUTTON_BOUNDS = (1325, 31, 1585, 82)


def draw_pose_overlay(
    frame: object,
    landmarks: list[dict[str, float]] | None,
    *,
    phase: str | None = None,
    shooting_hand: str = "right",
) -> object:
    """Draw a high-contrast pose skeleton on a camera frame."""
    import cv2

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    if landmarks:
        points = [_to_pixel(point, width, height) for point in landmarks]
        highlights = set(SIDE_LANDMARKS[shooting_hand].values())
        for start, end in POSE_CONNECTIONS:
            if (
                start >= len(points)
                or end >= len(points)
                or points[start] is None
                or points[end] is None
            ):
                continue
            color = ACCENT if start in highlights or end in highlights else GREEN
            cv2.line(annotated, points[start], points[end], (20, 20, 20), 6, cv2.LINE_AA)
            cv2.line(annotated, points[start], points[end], color, 3, cv2.LINE_AA)
        for index, point in enumerate(points):
            if point is not None:
                cv2.circle(annotated, point, 6, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(
                    annotated, point, 3, ACCENT if index in highlights else TEXT, -1, cv2.LINE_AA
                )
    if phase:
        _pill(annotated, 18, 18, f"PHASE  {phase.upper()}", ACCENT)
    return annotated


def compose_live_dashboard(
    frame: object,
    *,
    status: str,
    shot_count: int,
    latest_shot: Any | None = None,
    fps: float | None = None,
    shooting_hand: str = "right",
    armed: bool = False,
) -> object:
    """Place the camera feed inside a polished 16:9 coaching dashboard."""
    import cv2
    import numpy as np

    # Native Full HD output keeps substantially more camera detail.
    canvas_w, canvas_h, margin, gap, side_w = 1920, 1080, 32, 30, 507
    content_y, content_h = 120, 928
    video_w = canvas_w - margin * 2 - gap - side_w
    canvas = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)
    _text(canvas, "BASKET", (margin, 62), 1.04, TEXT, 2)
    brand_w = cv2.getTextSize("BASKET", cv2.FONT_HERSHEY_SIMPLEX, 1.04, 2)[0][0]
    _text(canvas, "VISION", (margin + brand_w + 7, 62), 1.04, ACCENT, 2)
    _text(canvas, "AI SHOOTING COACH", (margin, 91), 0.51, MUTED, 1)
    _record_button(canvas, armed)
    status_label, status_color = _status_details(status)
    _pill(canvas, canvas_w - 300, 34, status_label, status_color, scale=1.3)
    camera = _cover_resize(frame, video_w, content_h)
    canvas[content_y : content_y + content_h, margin : margin + video_w] = camera
    cv2.rectangle(
        canvas, (margin, content_y), (margin + video_w, content_y + content_h), (55, 51, 47), 1
    )
    _pill(canvas, margin + 24, content_y + 24, "LIVE", RED, scale=1.3)
    _pill(
        canvas,
        margin + 24,
        content_y + content_h - 64,
        f"{shooting_hand.upper()} HAND",
        ACCENT,
        scale=1.3,
    )
    if fps is not None:
        _text(
            canvas,
            f"{fps:.0f} FPS",
            (margin + video_w - 96, content_y + content_h - 32),
            0.56,
            MUTED,
            1,
        )
    panel = np.full((696, 380, 3), BG, dtype=np.uint8)
    _draw_score_panel(panel, 0, 0, 380, 696, shot_count, latest_shot, status)
    panel = cv2.resize(panel, (side_w, content_h), interpolation=cv2.INTER_CUBIC)
    panel_x = margin + video_w + gap
    canvas[content_y : content_y + content_h, panel_x : panel_x + side_w] = panel
    return canvas


def _draw_score_panel(
    canvas: object,
    x: int,
    y: int,
    width: int,
    height: int,
    shot_count: int,
    latest_shot: Any | None,
    status: str,
) -> None:
    import cv2

    _card(canvas, x, y, width, height)
    _text(canvas, "SHOT ANALYSIS", (x + 24, y + 36), 0.45, MUTED, 1)
    _text(canvas, f"SESSION  {shot_count:02d} SHOTS", (x + 196, y + 36), 0.38, MUTED, 1)
    if latest_shot is None:
        center_y = y + 225
        cv2.circle(canvas, (x + width // 2, center_y), 72, CARD_ALT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x + width // 2, center_y), 72, (61, 56, 51), 2, cv2.LINE_AA)
        _text(canvas, "--", (x + width // 2 - 30, center_y + 14), 1.25, MUTED, 2)
        _text(canvas, "READY FOR YOUR SHOT", (x + 75, center_y + 116), 0.48, TEXT, 1)
        instructions = (
            "Keep your full body visible",
            "Face sideways or at 45 degrees",
            "Shoot naturally when ready",
        )
        for index, line in enumerate(instructions, 1):
            row_y = center_y + 168 + (index - 1) * 46
            cv2.circle(canvas, (x + 34, row_y - 5), 12, CARD_ALT, -1)
            _text(canvas, str(index), (x + 30, row_y), 0.34, ACCENT, 1)
            _text(canvas, line, (x + 56, row_y), 0.41, MUTED, 1)
        _text(canvas, "Q  QUIT       R  RESET", (x + 92, y + height - 28), 0.38, MUTED, 1)
        return
    rating = latest_shot.rating
    overall = float(rating.get("overall", 0.0))
    grade, color = str(rating.get("grade", "Rated")), _score_color(overall)
    cv2.circle(canvas, (x + 84, y + 112), 48, CARD_ALT, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + 84, y + 112), (48, 48), -90, 0, overall * 3.6, color, 5)
    score_text = f"{overall:.0f}"
    score_w = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
    _text(canvas, score_text, (x + 84 - score_w // 2, y + 120), 0.9, TEXT, 2)
    _text(canvas, grade.upper(), (x + 154, y + 105), 0.74, color, 2)
    _text(canvas, f"SHOT #{latest_shot.shot_id}  /  100 POINTS", (x + 154, y + 132), 0.4, MUTED, 1)
    cv2.line(canvas, (x + 24, y + 177), (x + width - 24, y + 177), (55, 51, 47), 1)
    _text(canvas, "FORM BREAKDOWN", (x + 24, y + 211), 0.43, MUTED, 1)
    categories = rating.get("categories") or {}
    rows = (
        ("Set point", "set_point"),
        ("Elbow extension", "elbow_extension"),
        ("Arm elevation", "arm_elevation"),
        ("Leg load", "leg_load"),
        ("Guide hand", "guide_hand"),
        ("Release height", "release_height"),
    )
    row_y = y + 247
    for label, key in rows:
        _metric_bar(canvas, x + 24, row_y, width - 48, label, categories.get(key))
        row_y += 50
    # Keep enough room for two two-line coaching tips plus bottom padding.
    coaching_y = y + 566
    cv2.line(canvas, (x + 24, coaching_y - 20), (x + width - 24, coaching_y - 20), (55, 51, 47), 1)
    _text(canvas, "COACHING FOCUS", (x + 24, coaching_y + 9), 0.43, ACCENT, 1)
    tips = rating.get("tips") or ["Great repetition. Keep your motion consistent."]
    tip_y = coaching_y + 40
    for tip in tips[:2]:
        cv2.circle(canvas, (x + 30, tip_y - 5), 3, ACCENT, -1)
        for line in _wrap_text(str(tip), 39)[:2]:
            _text(canvas, line, (x + 43, tip_y), 0.39, TEXT, 1)
            tip_y += 20
        tip_y += 8


def _metric_bar(canvas: object, x: int, y: int, width: int, label: str, value: Any) -> None:
    import cv2

    numeric = float(value) if isinstance(value, int | float) else 0.0
    _text(canvas, label, (x, y), 0.41, TEXT, 1)
    _text(
        canvas,
        f"{numeric:.0f}" if isinstance(value, int | float) else "--",
        (x + width - 25, y),
        0.42,
        _score_color(numeric),
        1,
    )
    cv2.rectangle(canvas, (x, y + 13), (x + width, y + 20), CARD_ALT, -1)
    fill = int(width * max(0.0, min(numeric, 100.0)) / 100.0)
    if fill:
        cv2.rectangle(canvas, (x, y + 13), (x + fill, y + 20), _score_color(numeric), -1)


def _cover_resize(frame: object, target_w: int, target_h: int) -> object:
    import cv2

    height, width = frame.shape[:2]
    scale = max(target_w / width, target_h / height)
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
    crop_x, crop_y = (resized.shape[1] - target_w) // 2, (resized.shape[0] - target_h) // 2
    return resized[crop_y : crop_y + target_h, crop_x : crop_x + target_w]


def _card(frame: object, x: int, y: int, w: int, h: int) -> None:
    import cv2

    cv2.rectangle(frame, (x, y), (x + w, y + h), CARD, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (52, 48, 44), 1)


def _pill(
    frame: object,
    x: int,
    y: int,
    label: str,
    color: tuple[int, int, int],
    scale: float = 1.0,
) -> None:
    import cv2

    font_scale = 0.38 * scale
    width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
    width += round(30 * scale)
    height = round(27 * scale)
    cv2.rectangle(frame, (x, y), (x + width, y + height), CARD_ALT, -1)
    cv2.circle(frame, (x + round(12 * scale), y + height // 2), round(4 * scale), color, -1)
    _text(
        frame,
        label,
        (x + round(22 * scale), y + round(18 * scale)),
        font_scale,
        TEXT,
        1,
    )


def _record_button(frame: object, armed: bool) -> None:
    """Draw the click target that arms one jump-shot recording."""
    import cv2

    left, top, right, bottom = RECORD_BUTTON_BOUNDS
    color = RED if armed else ACCENT
    cv2.rectangle(frame, (left, top), (right, bottom), color, -1, cv2.LINE_AA)
    cv2.circle(frame, (left + 25, (top + bottom) // 2), 7, TEXT, -1, cv2.LINE_AA)
    label = "CANCEL RECORDING" if armed else "RECORD SHOT"
    _text(frame, label, (left + 45, top + 33), 0.55, BG, 2)


def _text(
    frame: object,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    import cv2

    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _status_details(status: str) -> tuple[str, tuple[int, int, int]]:
    return {
        "idle": ("CLICK RECORD", MUTED),
        "ready": ("READY", GREEN),
        "shooting": ("ANALYZING", ACCENT),
        "rated": ("SHOT RATED", GREEN),
        "cooldown": ("RESETTING", ORANGE),
        "no pose": ("BODY NOT FOUND", RED),
        "tracking": ("TRACKING", ACCENT),
    }.get(status, (status.upper(), MUTED))


def _wrap_text(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _score_color(score: float) -> tuple[int, int, int]:
    return GREEN if score >= 80 else ACCENT if score >= 60 else ORANGE if score >= 40 else RED


def save_frame(path: Path, frame: object) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def _to_pixel(point: dict[str, float], width: int, height: int) -> tuple[int, int] | None:
    if point.get("visibility", 1.0) < 0.35 or point.get("presence", 1.0) < 0.35:
        return None
    return int(point["x"] * width), int(point["y"] * height)
