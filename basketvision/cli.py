"""Command-line interface for local BasketVision analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from basketvision.analysis import analyze_video
from basketvision.camera import run_live_camera


def positive_int(value: str) -> int:
    """Parse a command-line integer that must be greater than zero."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_analyze_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a local basketball jump-shot video.",
        epilog="For live camera mode, run: python -m basketvision.cli live --help",
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
        type=positive_int,
        default=1,
        help="Process every Nth frame. Use >1 for faster development runs.",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
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


def build_live_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the webcam, draw body traces, detect jump shots, and rate form.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
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
        "--no-mirror",
        action="store_true",
        help="Disable mirrored preview (default preview is mirrored like a mirror).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args_list = list(sys.argv[1:] if argv is None else argv)

    if args_list and args_list[0] == "live":
        parser = build_live_parser()
        live_args = parser.parse_args(args_list[1:])
        try:
            run_live_camera(
                camera_index=live_args.camera,
                model_dir=live_args.model_dir,
                model_size=live_args.model_size,
                shooting_hand=live_args.shooting_hand,
                mirror=not live_args.no_mirror,
            )
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        return

    if args_list and args_list[0] == "analyze":
        args_list = args_list[1:]

    parser = build_analyze_parser()
    args = parser.parse_args(args_list)
    try:
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
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"Analysis complete: {output_dir}")


def main_live() -> None:
    """Console-script entry point for live camera mode."""
    main(["live", *sys.argv[1:]])


if __name__ == "__main__":
    main()
