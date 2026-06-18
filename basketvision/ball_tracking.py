"""Optional YOLO-based ball and hoop tracking.

This module is intentionally not part of the default MVP path. It is here so ball/hoop
tracking has a clear integration point once pose-only analysis is reliable.
"""

from __future__ import annotations

from pathlib import Path

from basketvision.video import iter_video_frames

DEFAULT_TARGET_LABELS = {
    "basketball",
    "sports ball",
    "ball",
    "hoop",
    "basketball hoop",
}


def track_ball_and_hoop(
    video_path: Path,
    *,
    model_name_or_path: str = "yolo11n.pt",
    confidence: float = 0.25,
    sample_every: int = 1,
    target_labels: set[str] | None = None,
    max_frames: int | None = None,
) -> dict[str, object]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO tracking requires the optional dependency: "
            'python -m pip install -e ".[yolo]"'
        ) from exc

    targets = {label.lower() for label in (target_labels or DEFAULT_TARGET_LABELS)}
    model = YOLO(model_name_or_path)
    frames = []

    for frame_index, timestamp_ms, frame in iter_video_frames(
        video_path, sample_every=sample_every, max_frames=max_frames
    ):
        result = model.predict(frame, conf=confidence, verbose=False)[0]
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = str(result.names[class_id]).lower()
            if label not in targets:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            detections.append(
                {
                    "label": label,
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                }
            )

        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "detections": detections,
            }
        )

    return {
        "model": model_name_or_path,
        "confidence": confidence,
        "target_labels": sorted(targets),
        "frames": frames,
    }

