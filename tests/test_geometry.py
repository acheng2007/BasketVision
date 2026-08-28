from basketvision.geometry import angle_degrees


def test_angle_degrees_returns_right_angle() -> None:
    a = {"x": 1.0, "y": 0.0, "z": 0.0}
    b = {"x": 0.0, "y": 0.0, "z": 0.0}
    c = {"x": 0.0, "y": 1.0, "z": 0.0}

    assert angle_degrees(a, b, c) == 90.0


def test_angle_degrees_returns_straight_line() -> None:
    a = {"x": -1.0, "y": 0.0, "z": 0.0}
    b = {"x": 0.0, "y": 0.0, "z": 0.0}
    c = {"x": 1.0, "y": 0.0, "z": 0.0}

    assert angle_degrees(a, b, c) == 180.0
