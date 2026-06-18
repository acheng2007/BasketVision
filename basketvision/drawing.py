"""Frame annotation helpers."""

from __future__ import annotations

from pathlib import Path

from basketvision.landmarks import POSE_CONNECTIONS, SIDE_LANDMARKS


def draw_pose_overlay(
    frame: object,
    landmarks: list[dict[str, float]] | None,
    *,
    phase: str | None = None,
    shooting_hand: str = "right",
) -> object:
    import cv2

    annotated = frame.copy()
    height, width = annotated.shape[:2]

    if landmarks:
        points = [_to_pixel(point, width, height) for point in landmarks]
        highlight_ids = set(SIDE_LANDMARKS[shooting_hand].values())

        for start, end in POSE_CONNECTIONS:
            if start >= len(points) or end >= len(points):
                continue
            if points[start] is None or points[end] is None:
                continue
            color = (0, 215, 255) if start in highlight_ids or end in highlight_ids else (80, 200, 120)
            cv2.line(annotated, points[start], points[end], color, 2)

        for idx, point in enumerate(points):
            if point is None:
                continue
            color = (0, 140, 255) if idx in highlight_ids else (255, 255, 255)
            cv2.circle(annotated, point, 4, color, -1)

    if phase:
        cv2.rectangle(annotated, (16, 16), (310, 58), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            f"Phase: {phase}",
            (28, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def save_frame(path: Path, frame: object) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def _to_pixel(point: dict[str, float], width: int, height: int) -> tuple[int, int] | None:
    visibility = point.get("visibility", 1.0)
    presence = point.get("presence", 1.0)
    if visibility < 0.35 or presence < 0.35:
        return None
    return int(point["x"] * width), int(point["y"] * height)

