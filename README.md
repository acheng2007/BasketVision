# BasketVision

BasketVision is a local-first basketball shooting form tracker. The first version analyzes a video file on your machine and writes all outputs to a local folder.

## Local MVP

The prototype does not use a database, web server, frontend, or cloud storage. The workflow is:

```text
local video -> OpenCV frames -> MediaPipe pose landmarks -> metrics/charts/annotated video
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

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

- Reads local video files with OpenCV.
- Extracts 33 MediaPipe pose landmarks per processed frame.
- Saves normalized and world landmarks to JSON.
- Computes core shooting-form angles.
- Estimates rough shot phases from pose motion.
- Draws skeleton overlays and key frames.
- Generates angle timeline charts and a Markdown report.

## Later

- Add a professional reference clip comparison.
- Add live webcam processing.
- Add a web UI only after the local analysis is useful.

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

