"""OpenCV video helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def read_video_metadata(video_path: Path) -> VideoMetadata:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    return VideoMetadata(
        path=str(video_path),
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        duration_seconds=frame_count / fps if fps else 0.0,
    )


def iter_video_frames(
    video_path: Path,
    *,
    sample_every: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, int, object]]:
    """Yield sampled BGR frames as `(frame_index, timestamp_ms, frame)`."""
    import cv2

    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_index = 0
    yielded = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % sample_every == 0:
            timestamp_ms = int((frame_index / fps) * 1000)
            yield frame_index, timestamp_ms, frame
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break

        frame_index += 1

    capture.release()


def create_video_writer(output_path: Path, fps: float, width: int, height: int):
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Could not create annotated video: {output_path}")
    return writer

