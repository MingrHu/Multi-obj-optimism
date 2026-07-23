"""ring_roundness 纯几何函数测试（合成数据，无需真实 KEY 文件）。"""

import numpy as np
import pytest

from mobo.extraction.ring_roundness import (
    algebraic_circle_center,
    closed_polyline_perimeter,
    fit_least_squares_circle,
    parse_float,
    polygon_signed_area,
    resample_closed_polyline,
)


def _circle_points(cx, cy, r, n=200):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))


def test_parse_float_fortran_d_exponent():
    assert parse_float("1.5D3") == pytest.approx(1500.0)
    assert parse_float("2.0e1") == pytest.approx(20.0)


def test_algebraic_circle_center():
    pts = _circle_points(3.0, -2.0, 5.0)
    center = algebraic_circle_center(pts)
    np.testing.assert_allclose(center, [3.0, -2.0], atol=1e-6)


def test_fit_least_squares_circle_perfect_circle():
    pts = _circle_points(0.0, 0.0, 10.0)
    res = fit_least_squares_circle(pts)
    assert res.method == "LSC"
    assert res.radius == pytest.approx(10.0, rel=1e-4)
    # 完美圆的圆度（Rmax - Rmin）≈ 0
    assert res.roundness == pytest.approx(0.0, abs=1e-4)


def test_polygon_signed_area_unit_square():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert polygon_signed_area(square) == pytest.approx(1.0)


def test_closed_polyline_perimeter_unit_square():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert closed_polyline_perimeter(square) == pytest.approx(4.0)


def test_resample_closed_polyline_count():
    pts = _circle_points(0.0, 0.0, 5.0, n=40)
    resampled = resample_closed_polyline(pts, 100)
    assert resampled.shape == (100, 2)
    # 重采样点仍应落在半径 5 附近
    radii = np.linalg.norm(resampled, axis=1)
    assert np.allclose(radii, 5.0, atol=0.1)


def test_resample_too_few_samples_raises():
    pts = _circle_points(0.0, 0.0, 5.0, n=40)
    with pytest.raises(ValueError):
        resample_closed_polyline(pts, 8)
