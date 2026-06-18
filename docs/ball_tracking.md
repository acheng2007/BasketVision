# Optional Ball And Hoop Tracking

The MVP is pose-first. YOLO ball/hoop tracking is optional and disabled by default.

## Why It Comes Later

Pose landmarks are enough to build the first mechanics pipeline: joint angles, shot phases, annotated playback, and reports. Ball/hoop tracking adds value after that because it can improve:

- release timing
- ball path visualization
- make/miss detection
- shot phase detection
- comparison against pro reference clips

## Enable Later

Install the optional dependency:

```bash
python -m pip install -e ".[yolo]"
```

Run analysis with YOLO enabled:

```bash
python scripts/analyze_video.py data/uploads/user-shot.mp4 --enable-ball-tracking
```

This writes `ball_tracking.json` into the output folder.

For best results, use a custom model trained to detect:

- basketball
- rim/hoop
- backboard, if useful

