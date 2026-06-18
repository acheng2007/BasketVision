# Analysis Visualizations

Each run writes a local output folder under `outputs/<video-name>/`.

## Annotated Playback

`annotated.mp4` overlays the detected pose skeleton onto the video. The selected shooting side is highlighted in orange, while the rest of the body skeleton is green/white. When phase detection succeeds, the video also shows the current phase label.

## Key Frames

`keyframes/` contains still images for the rough shot phases:

- gather
- set point
- release
- follow-through
- landing

These frames are the fastest way to inspect whether the phase detector found sensible moments.

## Charts

`charts/` contains one timeline chart per metric. These are local PNG files that show how angles change over the shot.

The most useful first charts are:

- shooting elbow angle
- shooting shoulder angle
- shooting knee angle
- release height proxy

## Report

`report.md` summarizes video metadata, detected phase key frames, metric ranges, and first-pass coaching notes.

