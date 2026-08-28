# BasketVision

BasketVision is a local-first basketball shooting form tracker. Analyze a video file, or open your webcam for live body tracing, jump-shot detection, and form rating.

## Local MVP

The prototype does not use a database, web server, frontend, or cloud storage.

```text
camera/video -> OpenCV frames -> MediaPipe pose landmarks -> metrics / rating / annotated output
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Live Camera (jump-shot rating)

Open a GUI window with your webcam, body skeleton overlay, automatic jump-shot detection, and a 0-100 form score:

```bash
python scripts/live_camera.py
```

Or:

```bash
basketvision-live
# equivalent:
python -m basketvision.cli live
```

Useful flags:

```bash
python -m basketvision.cli live --camera 0 --shooting-hand right
python -m basketvision.cli live --no-mirror --shooting-hand left
```

Controls:

- `q` / `Esc` - quit
- `Space` - arm or cancel shot recording
- `r` - reset detector / clear last rating

Stand mostly sideways or at ~45° so the shooting arm is visible, then take a normal jump shot. When the wrist rises and the shot finishes, BasketVision rates set point, elbow extension, leg load, guide hand, and release height.

## Analyze A Video

Put a jump-shot video in `data/uploads/`, then run:

```bash
python scripts/analyze_video.py data/uploads/user-shot.mp4
```

The script creates:

```text
outputs/user-shot/
  landmarks.json
  metrics.json
  phases.json
  keyframes/
  charts/
  annotated.mp4
  report.md
```

## Current Capabilities

- Live webcam GUI with skeleton overlay and on-screen form rating.
- Automatic jump-shot detection from shooting-wrist rise/fall.
- Reads local video files with OpenCV.
- Extracts 33 MediaPipe pose landmarks per processed frame.
- Saves normalized and world landmarks to JSON.
- Computes core shooting-form angles.
- Estimates rough shot phases from pose motion.
- Draws skeleton overlays and key frames.
- Generates angle timeline charts and a Markdown report.

## Later

- Add a professional reference clip comparison.
- Add a web UI only after the local analysis is useful.
- Tighten live detection with ball tracking / knee extension cues.

## Optional YOLO Ball Tracking

Ball/hoop tracking is disabled by default. Once pose-only analysis is reliable, install the optional YOLO dependency:

```bash
python -m pip install -e ".[yolo]"
```

Then run:

```bash
python scripts/analyze_video.py data/uploads/user-shot.mp4 --enable-ball-tracking
```

This creates `ball_tracking.json` in the output folder.
