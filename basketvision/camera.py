"""Live webcam GUI with pose overlay, shot detection, and form rating."""

from __future__ import annotations

import time
from pathlib import Path

from basketvision.drawing import RECORD_BUTTON_BOUNDS, compose_live_dashboard, draw_pose_overlay
from basketvision.live_detector import DetectedShot, ShotDetector
from basketvision.pose import PoseExtractor, ensure_pose_model


def run_live_camera(
    *,
    camera_index: int = 0,
    model_dir: Path = Path("models"),
    model_size: str = "lite",
    shooting_hand: str = "right",
    mirror: bool = True,
    window_name: str = "BasketVision Live",
) -> None:
    """Open the camera, draw body traces, detect jump shots, and show ratings."""
    import cv2

    model_path = ensure_pose_model(model_dir, model_size=model_size)

    # Mirrored preview swaps image left/right, so MediaPipe's hand labels flip too.
    analysis_hand = ("left" if shooting_hand == "right" else "right") if mirror else shooting_hand
    detector = ShotDetector(shooting_hand=analysis_hand)
    latest_shot: DetectedShot | None = None

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check permissions and try another --camera value."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)

    control = {"armed": False, "toggle_requested": False}

    def handle_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        left, top, right, bottom = RECORD_BUTTON_BOUNDS
        if left <= x <= right and top <= y <= bottom:
            control["toggle_requested"] = True

    cv2.setMouseCallback(window_name, handle_mouse)

    print("BasketVision Live")
    print("  q / Esc  quit")
    print("  click RECORD SHOT to arm one attempt")
    print("  space    arm / cancel recording")
    print("  r        reset last rating")
    print("Stand sideways / 45° to the camera and take a jump shot.")

    frame_index = 0
    started = time.perf_counter()
    fps_started = started
    fps_frame_count = 0
    displayed_fps = 0.0
    last_timestamp_ms = 0
    try:
        with PoseExtractor(model_path) as extractor:
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("Camera frame grab failed - exiting.")
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int((time.perf_counter() - started) * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                pose = extractor.process_frame(frame, timestamp_ms)
                landmarks = pose["landmarks"] if pose else []

                if control["toggle_requested"]:
                    control["toggle_requested"] = False
                    control["armed"] = not control["armed"]
                    detector.reset()
                    message = "Shot recording armed." if control["armed"] else "Cancelled."
                    print(message)

                detected = None
                if control["armed"]:
                    detected = detector.update(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        landmarks=landmarks,
                    )
                if detected is not None:
                    latest_shot = detected
                    control["armed"] = False
                    print(
                        f"Shot #{detected.shot_id}: "
                        f"{detected.rating['overall']:.0f}/100 "
                        f"({detected.rating['grade']})"
                    )
                    for tip in detected.rating.get("tips", []):
                        print(f"  - {tip}")

                fps_frame_count += 1
                fps_elapsed = time.perf_counter() - fps_started
                if fps_elapsed >= 0.5:
                    displayed_fps = fps_frame_count / fps_elapsed
                    fps_frame_count = 0
                    fps_started = time.perf_counter()

                annotated = draw_pose_overlay(
                    frame,
                    landmarks if landmarks else None,
                    shooting_hand=analysis_hand,
                )
                dashboard_status = detector.status if control["armed"] else "idle"
                if latest_shot is not None and not control["armed"]:
                    dashboard_status = "rated"
                annotated = compose_live_dashboard(
                    annotated,
                    status=dashboard_status,
                    shot_count=detector.shot_count,
                    latest_shot=latest_shot,
                    fps=displayed_fps,
                    shooting_hand=shooting_hand,
                    armed=control["armed"],
                )

                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    latest_shot = None
                    control["armed"] = False
                    detector.reset()
                    print("Reset ratings / detector.")
                if key == ord(" "):
                    control["toggle_requested"] = True

                frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()
