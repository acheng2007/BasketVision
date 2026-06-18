"""MediaPipe Pose Landmarker integration."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def ensure_pose_model(model_dir: Path, model_size: str = "lite") -> Path:
    if model_size not in MODEL_URLS:
        valid = ", ".join(sorted(MODEL_URLS))
        raise ValueError(f"Unknown model size '{model_size}'. Expected one of: {valid}")

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"pose_landmarker_{model_size}.task"
    if model_path.exists():
        return model_path

    print(f"Downloading MediaPipe pose model to {model_path}...")
    urlretrieve(MODEL_URLS[model_size], model_path)
    return model_path


class PoseExtractor:
    """Thin wrapper around MediaPipe's video-mode Pose Landmarker."""

    def __init__(self, model_path: Path, *, num_poses: int = 1) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            output_segmentation_masks=False,
        )
        self._mp = mp
        self._vision = vision
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def process_frame(self, frame_bgr: object, timestamp_ms: int) -> dict[str, object] | None:
        import cv2
        import numpy as np

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        landmarks = [_landmark_to_dict(point) for point in result.pose_landmarks[0]]
        world_landmarks = []
        if result.pose_world_landmarks:
            world_landmarks = [_landmark_to_dict(point) for point in result.pose_world_landmarks[0]]

        return {
            "landmarks": landmarks,
            "world_landmarks": world_landmarks,
        }


def _landmark_to_dict(point: object) -> dict[str, float]:
    values = {
        "x": float(point.x),
        "y": float(point.y),
        "z": float(point.z),
    }
    visibility = getattr(point, "visibility", None)
    presence = getattr(point, "presence", None)
    if visibility is not None:
        values["visibility"] = float(visibility)
    if presence is not None:
        values["presence"] = float(presence)
    return values

