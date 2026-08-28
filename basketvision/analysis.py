"""End-to-end local video analysis pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

from basketvision.ball_tracking import track_ball_and_hoop
from basketvision.charts import write_metric_charts
from basketvision.drawing import draw_pose_overlay, save_frame
from basketvision.metrics import compute_metrics
from basketvision.phases import detect_phases, phase_lookup
from basketvision.pose import PoseExtractor, ensure_pose_model
from basketvision.report import write_report
from basketvision.video import create_video_writer, iter_video_frames, read_video_metadata


def analyze_video(
    video_path: Path,
    *,
    output_root: Path = Path("outputs"),
    model_dir: Path = Path("models"),
    model_size: str = "lite",
    shooting_hand: str = "right",
    sample_every: int = 1,
    max_frames: int | None = None,
    enable_ball_tracking: bool = False,
    yolo_model: str = "yolo11n.pt",
) -> Path:
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = output_root / video_path.stem
    keyframe_dir = output_dir / "keyframes"
    chart_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_visuals(keyframe_dir, chart_dir)

    metadata = read_video_metadata(video_path)
    model_path = ensure_pose_model(model_dir, model_size=model_size)

    landmark_frames = []
    with PoseExtractor(model_path) as extractor:
        for frame_index, timestamp_ms, frame in iter_video_frames(
            video_path, sample_every=sample_every, max_frames=max_frames
        ):
            pose = extractor.process_frame(frame, timestamp_ms)
            landmark_frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "landmarks": pose["landmarks"] if pose else [],
                    "world_landmarks": pose["world_landmarks"] if pose else [],
                }
            )

    metrics = compute_metrics(landmark_frames, shooting_hand=shooting_hand)
    phases = detect_phases(metrics)
    chart_paths = write_metric_charts(metrics, chart_dir)

    _write_json(
        output_dir / "landmarks.json",
        {
            "video": metadata.to_dict(),
            "model": {"name": "mediapipe_pose_landmarker", "size": model_size},
            "frames": landmark_frames,
        },
    )
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "phases.json", phases)

    if enable_ball_tracking:
        _write_json(
            output_dir / "ball_tracking.json",
            track_ball_and_hoop(
                video_path,
                model_name_or_path=yolo_model,
                sample_every=sample_every,
                max_frames=max_frames,
            ),
        )

    _write_annotated_video_and_keyframes(
        video_path=video_path,
        output_dir=output_dir,
        metadata=metadata.to_dict(),
        landmark_frames=landmark_frames,
        phases=phases,
        shooting_hand=shooting_hand,
        sample_every=sample_every,
        max_frames=max_frames,
    )

    write_report(
        output_dir / "report.md",
        video_path=video_path,
        metadata=metadata.to_dict(),
        metrics=metrics,
        phases=phases,
        chart_paths=chart_paths,
        ball_tracking_enabled=enable_ball_tracking,
    )

    return output_dir


def _write_annotated_video_and_keyframes(
    *,
    video_path: Path,
    output_dir: Path,
    metadata: dict[str, object],
    landmark_frames: list[dict[str, object]],
    phases: dict[str, object],
    shooting_hand: str,
    sample_every: int,
    max_frames: int | None,
) -> None:
    frame_landmarks = {
        int(frame["frame_index"]): frame.get("landmarks") or [] for frame in landmark_frames
    }
    phase_by_frame = phase_lookup(phases)
    keyframe_refs = phases.get("keyframes", {})
    keyframe_indices = set()
    if isinstance(keyframe_refs, dict):
        keyframe_indices = {
            int(ref["frame_index"]) for ref in keyframe_refs.values() if isinstance(ref, dict)
        }

    output_fps = max(float(metadata["fps"]) / sample_every, 1.0)
    writer = create_video_writer(
        output_dir / "annotated.mp4",
        output_fps,
        int(metadata["width"]),
        int(metadata["height"]),
    )

    try:
        for frame_index, _timestamp_ms, frame in iter_video_frames(
            video_path, sample_every=sample_every, max_frames=max_frames
        ):
            landmarks = frame_landmarks.get(frame_index, [])
            phase = phase_by_frame.get(frame_index)
            annotated = draw_pose_overlay(
                frame,
                landmarks,
                phase=phase,
                shooting_hand=shooting_hand,
            )
            writer.write(annotated)

            if frame_index in keyframe_indices:
                label = phase or f"frame_{frame_index}"
                save_frame(output_dir / "keyframes" / f"{label}.jpg", annotated)
    finally:
        writer.release()


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2) + "\n", encoding="utf-8")


def _remove_stale_visuals(keyframe_dir: Path, chart_dir: Path) -> None:
    """Remove visuals generated by an earlier run for the same video name."""
    for directory, pattern in ((keyframe_dir, "*.jpg"), (chart_dir, "*.png")):
        for path in directory.glob(pattern):
            path.unlink()


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
