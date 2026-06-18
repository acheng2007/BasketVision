"""Command-line interface for local BasketVision analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from basketvision.analysis import analyze_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a local basketball jump-shot video.",
    )
    parser.add_argument("video", type=Path, help="Path to a local video file.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Folder where analysis output directories are written.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Folder for the MediaPipe .task model file.",
    )
    parser.add_argument(
        "--model-size",
        choices=["lite", "full", "heavy"],
        default="lite",
        help="MediaPipe Pose Landmarker model size.",
    )
    parser.add_argument(
        "--shooting-hand",
        choices=["left", "right"],
        default="right",
        help="Which arm to treat as the shooting arm.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Process every Nth frame. Use >1 for faster development runs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for quick test runs.",
    )
    parser.add_argument(
        "--enable-ball-tracking",
        action="store_true",
        help="Also run optional YOLO ball/hoop detection and save ball_tracking.json.",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo11n.pt",
        help="YOLO model name or path for optional ball/hoop tracking.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = analyze_video(
        args.video,
        output_root=args.output_root,
        model_dir=args.model_dir,
        model_size=args.model_size,
        shooting_hand=args.shooting_hand,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        enable_ball_tracking=args.enable_ball_tracking,
        yolo_model=args.yolo_model,
    )
    print(f"Analysis complete: {output_dir}")


if __name__ == "__main__":
    main()

