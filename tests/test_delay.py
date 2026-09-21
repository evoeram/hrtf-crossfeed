"""Тесты delay: интерпретация dimensions, единицы измерения."""

import numpy as np

from hrtf_crossfeed.delay import (
    delay_to_seconds,
    delay_value,
    get_delay_seconds,
)


def test_delay_none():
    assert delay_value(None, 0, 0, 10, 2) == 0.0
    assert get_delay_seconds(None, 0, 0, 10, 2, 48000.0) == 0.0


def test_delay_scalar():
    info = {
        "data": np.array(5.0),
        "dimensions": [],
        "units": "sample",
    }
    assert delay_value(info, 0, 0, 10, 2) == 5.0


def test_delay_per_measurement_per_ear():
    """Data.Delay shape [M, R] = [3, 2]."""
    d = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    info = {
        "data": d,
        "dimensions": ["M", "R"],
        "units": "sample",
    }

    assert delay_value(info, 0, 0, 3, 2) == 1.0
    assert delay_value(info, 0, 1, 3, 2) == 2.0
    assert delay_value(info, 1, 0, 3, 2) == 3.0
    assert delay_value(info, 2, 1, 3, 2) == 6.0


def test_delay_single_value():
    """Data.Delay shape [1] — одна задержка для всех."""
    info = {
        "data": np.array([42.0]),
        "dimensions": ["M"],
        "units": "sample",
    }
    assert delay_value(info, 5, 1, 10, 2) == 42.0


def test_delay_to_seconds_sample():
    assert np.isclose(delay_to_seconds(480.0, "sample", 48000.0), 0.01)


def test_delay_to_seconds_second():
    assert np.isclose(delay_to_seconds(0.01, "second", 48000.0), 0.01)


def test_delay_to_seconds_zero():
    """Нулевая задержка — единицы не важны."""
    assert delay_to_seconds(0.0, "", 48000.0) == 0.0


def test_delay_to_seconds_unknown_units():
    """Неизвестные единицы — предупреждение, предполагаются секунды."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isclose(delay_to_seconds(0.005, "foo", 48000.0), 0.005)


def test_delay_to_seconds_empty_units():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isclose(delay_to_seconds(0.003, "", 48000.0), 0.003)


def test_get_delay_seconds_full():
    info = {
        "data": np.array([
            [100.0, 200.0],
        ]),
        "dimensions": ["M", "R"],
        "units": "sample",
    }

    fs = 48000.0

    assert np.isclose(
        get_delay_seconds(info, 0, 0, 1, 2, fs),
        100.0 / fs,
    )
    assert np.isclose(
        get_delay_seconds(info, 0, 1, 1, 2, fs),
        200.0 / fs,
    )
