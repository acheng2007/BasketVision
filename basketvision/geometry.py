"""Geometry utilities for pose-derived measurements."""

from __future__ import annotations

import math
from collections.abc import Sequence


def angle_degrees(
    a: dict[str, float],
    b: dict[str, float],
    c: dict[str, float],
) -> float:
    """Return angle ABC in degrees."""
    bax = a["x"] - b["x"]
    bay = a["y"] - b["y"]
    baz = a.get("z", 0.0) - b.get("z", 0.0)

    bcx = c["x"] - b["x"]
    bcy = c["y"] - b["y"]
    bcz = c.get("z", 0.0) - b.get("z", 0.0)

    dot = bax * bcx + bay * bcy + baz * bcz
    mag_ba = math.sqrt(bax**2 + bay**2 + baz**2)
    mag_bc = math.sqrt(bcx**2 + bcy**2 + bcz**2)
    if mag_ba == 0 or mag_bc == 0:
        return math.nan

    cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine))


def midpoint(points: Sequence[dict[str, float]]) -> dict[str, float]:
    count = len(points)
    return {
        "x": sum(point["x"] for point in points) / count,
        "y": sum(point["y"] for point in points) / count,
        "z": sum(point.get("z", 0.0) for point in points) / count,
    }


def landmark_is_usable(point: dict[str, float], threshold: float = 0.35) -> bool:
    return point.get("visibility", 1.0) >= threshold and point.get("presence", 1.0) >= threshold
