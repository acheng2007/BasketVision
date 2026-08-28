"""Live jump-shot detection from a rolling pose-metric buffer."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from basketvision.metrics import compute_frame_metrics, compute_metrics
from basketvision.phases import detect_phases
from basketvision.rating import rate_shot


@dataclass
class DetectedShot:
    """A completed jump-shot segment with rating."""

    shot_id: int
    start_frame: int
    end_frame: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    rating: dict[str, Any]
    phases: dict[str, Any]
    metrics: dict[str, Any]


@dataclass
class ShotDetector:
    """Stateful detector that watches live frame metrics for jump-shot motion."""

    shooting_hand: str = "right"
    pre_roll_ms: int = 700
    post_roll_ms: int = 450
    min_shot_ms: int = 350
    max_shot_ms: int = 2500
    rise_threshold: float = 0.045
    peak_min_height: float = 0.48
    cooldown_ms: int = 900
    history_ms: int = 4000

    _buffer: deque[dict[str, Any]] = field(default_factory=deque, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)
    _rise_start_ms: int | None = field(default=None, init=False, repr=False)
    _peak_height: float = field(default=0.0, init=False, repr=False)
    _peak_ms: int | None = field(default=None, init=False, repr=False)
    _baseline_height: float | None = field(default=None, init=False, repr=False)
    _cooldown_until_ms: int = field(default=0, init=False, repr=False)
    _shot_count: int = field(default=0, init=False, repr=False)
    _last_status: str = field(default="ready", init=False, repr=False)

    def reset(self) -> None:
        self._buffer.clear()
        self._active = False
        self._rise_start_ms = None
        self._peak_height = 0.0
        self._peak_ms = None
        self._baseline_height = None
        self._cooldown_until_ms = 0
        self._last_status = "ready"

    @property
    def status(self) -> str:
        return self._last_status

    @property
    def shot_count(self) -> int:
        return self._shot_count

    def update(
        self,
        *,
        frame_index: int,
        timestamp_ms: int,
        landmarks: list[dict[str, float]],
    ) -> DetectedShot | None:
        if not landmarks:
            self._last_status = "no pose"
            return None

        frame_metrics = compute_frame_metrics(
            landmarks,
            shooting_hand=self.shooting_hand,
            timestamp_ms=timestamp_ms,
        )
        frame_metrics["frame_index"] = frame_index
        frame_metrics["landmarks"] = landmarks
        self._buffer.append(frame_metrics)
        self._trim_history(timestamp_ms)

        # Start detection watches both wrists. This is robust to mirrored camera
        # previews and to two-handed gathers, while form scoring still uses the
        # configured shooting side.
        height = _highest_visible_wrist(landmarks)
        if height is None:
            self._last_status = "tracking"
            return None

        if timestamp_ms < self._cooldown_until_ms:
            self._last_status = "cooldown"
            self._baseline_height = height
            return None

        if not self._active:
            return self._maybe_start_shot(timestamp_ms, height)

        return self._continue_shot(timestamp_ms, height)

    def _maybe_start_shot(self, timestamp_ms: int, height: float) -> DetectedShot | None:
        if self._baseline_height is None:
            self._baseline_height = height
            self._last_status = "ready"
            return None

        # Follow a lowered hand reasonably quickly, but adapt upward very slowly.
        # Otherwise a smooth, gradual gather can move the baseline with the wrist
        # and never cross the shot-start threshold.
        adaptation = 0.20 if height < self._baseline_height else 0.005
        self._baseline_height = ((1.0 - adaptation) * self._baseline_height) + adaptation * height
        rise = height - self._baseline_height

        if rise >= self.rise_threshold and height >= self.peak_min_height * 0.85:
            self._active = True
            self._rise_start_ms = timestamp_ms
            self._peak_height = height
            self._peak_ms = timestamp_ms
            self._last_status = "shooting"
            return None

        self._last_status = "ready"
        return None

    def _continue_shot(self, timestamp_ms: int, height: float) -> DetectedShot | None:
        assert self._rise_start_ms is not None
        assert self._peak_ms is not None

        if height > self._peak_height:
            self._peak_height = height
            self._peak_ms = timestamp_ms

        elapsed = timestamp_ms - self._rise_start_ms
        since_peak = timestamp_ms - self._peak_ms
        dropped = self._peak_height - height

        finished = False
        if since_peak >= self.post_roll_ms and dropped >= self.rise_threshold * 0.45:
            finished = True
        elif elapsed >= self.max_shot_ms:
            finished = True

        self._last_status = "shooting"
        if not finished:
            return None

        shot = self._finalize_shot(end_timestamp_ms=timestamp_ms)
        self._active = False
        self._rise_start_ms = None
        self._peak_ms = None
        self._peak_height = 0.0
        self._baseline_height = height
        self._cooldown_until_ms = timestamp_ms + self.cooldown_ms
        self._last_status = "rated" if shot else "ready"
        return shot

    def _finalize_shot(self, *, end_timestamp_ms: int) -> DetectedShot | None:
        assert self._rise_start_ms is not None

        if self._peak_height < self.peak_min_height:
            return None

        start_ms = max(0, self._rise_start_ms - self.pre_roll_ms)
        end_ms = end_timestamp_ms
        if end_ms - start_ms < self.min_shot_ms:
            return None

        segment = [
            frame for frame in self._buffer if start_ms <= int(frame["timestamp_ms"]) <= end_ms
        ]
        if len(segment) < 8:
            return None

        landmark_frames = [
            {
                "frame_index": int(frame["frame_index"]),
                "timestamp_ms": int(frame["timestamp_ms"]),
                "landmarks": frame.get("landmarks") or [],
            }
            for frame in segment
        ]
        metrics = compute_metrics(landmark_frames, shooting_hand=self.shooting_hand)
        phases = detect_phases(metrics)
        rating = rate_shot(metrics, phases)

        self._shot_count += 1
        return DetectedShot(
            shot_id=self._shot_count,
            start_frame=int(segment[0]["frame_index"]),
            end_frame=int(segment[-1]["frame_index"]),
            start_timestamp_ms=int(segment[0]["timestamp_ms"]),
            end_timestamp_ms=int(segment[-1]["timestamp_ms"]),
            rating=rating,
            phases=phases,
            metrics=metrics,
        )

    def _trim_history(self, timestamp_ms: int) -> None:
        cutoff = timestamp_ms - self.history_ms
        while self._buffer and int(self._buffer[0]["timestamp_ms"]) < cutoff:
            self._buffer.popleft()


def _highest_visible_wrist(landmarks: list[dict[str, float]]) -> float | None:
    """Return the normalized height of the highest reliably visible wrist."""
    heights = []
    for index in (15, 16):
        if index >= len(landmarks):
            continue
        wrist = landmarks[index]
        if wrist.get("visibility", 1.0) < 0.35 or wrist.get("presence", 1.0) < 0.35:
            continue
        y = float(wrist.get("y", math.nan))
        if math.isfinite(y):
            heights.append(1.0 - y)
    return max(heights) if heights else None
