# Shot Metrics

The local prototype starts with deterministic pose metrics before training any model.

## Core Measurements

- **Shooting elbow angle:** shoulder-elbow-wrist angle for the selected shooting hand.
- **Shooting shoulder angle:** hip-shoulder-elbow angle for the selected shooting hand.
- **Shooting hip angle:** shoulder-midpoint to hip-midpoint to shooting-side knee.
- **Shooting knee angle:** hip-knee-ankle angle for the selected shooting side.
- **Guide elbow angle:** shoulder-elbow-wrist angle for the non-shooting arm.
- **Release height proxy:** `1 - wrist_y`, where MediaPipe `y` grows downward.

## First-Pass Phase Detection

The current phase detector estimates release as the frame where the shooting wrist is highest. It then labels surrounding frames as:

- `gather`
- `set_point`
- `release`
- `follow_through`
- `landing`

This is intentionally simple. Once we test real clips, we can improve it with ball tracking, velocity, knee extension, or manual labels.

