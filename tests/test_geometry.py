"""Тесты geometry: преобразование координат, поиск ближайшего."""

import numpy as np

from hrtf_crossfeed.geometry import (
    direction_to_unit_vector,
    nearest_source_index,
    source_positions_to_unit_vectors,
)


def test_spherical_basic():
    """Сферические координаты → единичные векторы."""
    # [az, el, radius]
    src = np.array([
        [0.0, 0.0, 1.0],
        [90.0, 0.0, 1.0],
        [0.0, 90.0, 1.0],
    ])

    vectors, az, el = source_positions_to_unit_vectors(
        src, position_type="spherical", units="degree, degree, metre"
    )

    # az=0, el=0 → (1, 0, 0)
    assert np.allclose(vectors[0], [1.0, 0.0, 0.0])
    # az=90, el=0 → (0, 1, 0)
    assert np.allclose(vectors[1], [0.0, 1.0, 0.0])
    # az=0, el=90 → (0, 0, 1)
    assert np.allclose(vectors[2], [0.0, 0.0, 1.0])

    assert np.allclose(az, [0.0, 90.0, 0.0])
    assert np.allclose(el, [0.0, 0.0, 90.0])


def test_cartesian_basic():
    src = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],  # zero vector — should raise
    ])

    # Первые три должны работать
    vectors, az, el = source_positions_to_unit_vectors(
        src[:3], position_type="cartesian", units=""
    )

    assert np.allclose(vectors[0], [1.0, 0.0, 0.0])
    assert np.allclose(vectors[1], [0.0, 1.0, 0.0])
    assert np.allclose(vectors[2], [0.0, 0.0, 1.0])

    assert np.isclose(az[0], 0.0)
    assert np.isclose(az[1], 90.0)
    assert np.isclose(el[2], 90.0)


def test_cartesian_zero_vector_raises():
    src = np.array([[0.0, 0.0, 0.0]])

    try:
        source_positions_to_unit_vectors(
            src, position_type="cartesian", units=""
        )
        assert False, "Should have raised"
    except ValueError:
        pass


def test_radian_units():
    src = np.array([
        [np.pi / 2, 0.0, 1.0],
    ])

    vectors, az, el = source_positions_to_unit_vectors(
        src, position_type="spherical", units="radian, radian, metre"
    )

    assert np.allclose(vectors[0], [0.0, 1.0, 0.0])
    assert np.isclose(az[0], 90.0)


def test_direction_to_unit_vector():
    v = direction_to_unit_vector(0.0, 0.0)
    assert np.allclose(v, [1.0, 0.0, 0.0])

    v = direction_to_unit_vector(90.0, 0.0)
    assert np.allclose(v, [0.0, 1.0, 0.0])

    v = direction_to_unit_vector(0.0, 90.0)
    assert np.allclose(v, [0.0, 0.0, 1.0])


def test_nearest_source_index():
    """Поиск ближайшего измерения к заданному направлению."""
    vectors = np.array([
        [1.0, 0.0, 0.0],   # az=0
        [0.0, 1.0, 0.0],   # az=90
        [-1.0, 0.0, 0.0],  # az=180
        [0.0, -1.0, 0.0],  # az=-90
    ])

    idx, err = nearest_source_index(vectors, 0.0, 0.0)
    assert idx == 0
    assert err < 1e-10

    idx, err = nearest_source_index(vectors, 90.0, 0.0)
    assert idx == 1
    assert err < 1e-10

    idx, err = nearest_source_index(vectors, -90.0, 0.0)
    assert idx == 3
    assert err < 1e-10


def test_nearest_source_index_interpolated():
    """Ближайший к азимуту 45° — между 0° и 90°."""
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    idx, err = nearest_source_index(vectors, 45.0, 0.0)
    # 45° — ровно между, но searchsorted вернёт первый
    assert idx in (0, 1)
    assert np.isclose(err, 45.0, atol=1.0)
