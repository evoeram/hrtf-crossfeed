"""Тесты smoothing: сохранение phase, disabled, limit gain."""

import numpy as np

from hrtf_crossfeed.smoothing import (
    limit_complex_magnitude,
    smooth_log_magnitude_octave,
)


def test_smooth_preserves_phase():
    """Сглаживание magnitude не должно менять phase (кроме DC/Nyquist)."""
    n = 256
    fs = 48000.0
    freqs = np.fft.rfftfreq(n, 1.0 / fs)

    rng = np.random.default_rng(123)
    mag = 1.0 + 0.5 * np.abs(rng.standard_normal(len(freqs)))
    phase = rng.uniform(-np.pi, np.pi, len(freqs))
    H = mag * np.exp(1j * phase)

    smoothed = smooth_log_magnitude_octave(H, freqs, fraction=12.0)

    # DC и Nyquist обнуляются (imag=0) — исключаем из проверки
    inner = slice(1, -1)
    assert np.allclose(np.angle(smoothed[inner]), phase[inner], atol=1e-10), \
        "Phase changed after magnitude-only smoothing"


def test_smooth_disabled_returns_copy():
    """При fraction=0 возвращается копия исходного массива."""
    H = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
    freqs = np.array([0.0, 100.0, 200.0])

    result = smooth_log_magnitude_octave(H, freqs, fraction=0.0)

    assert np.allclose(result, H)
    assert result is not H  # должен быть копией


def test_smooth_none_returns_copy():
    H = np.array([1 + 1j, 2 + 2j], dtype=np.complex128)
    freqs = np.array([0.0, 100.0])

    result = smooth_log_magnitude_octave(H, freqs, fraction=None)

    assert np.allclose(result, H)
    assert result is not H


def test_smooth_reduces_variance():
    """Сглаживание должно уменьшать дисперсию magnitude."""
    n = 512
    fs = 48000.0
    freqs = np.fft.rfftfreq(n, 1.0 / fs)

    rng = np.random.default_rng(42)
    mag = np.exp(rng.standard_normal(len(freqs)) * 0.5)
    H = mag.astype(np.complex128)

    smoothed = smooth_log_magnitude_octave(H, freqs, fraction=6.0)

    # Только для положительных частот
    pos = freqs > 0
    var_orig = np.var(np.log(np.abs(H[pos]) + 1e-12))
    var_smooth = np.var(np.log(np.abs(smoothed[pos]) + 1e-12))

    assert var_smooth < var_orig, \
        f"Variance not reduced: {var_smooth} >= {var_orig}"


def test_smooth_dc_nyquist_real():
    """DC и Nyquist должны быть действительными после сглаживания."""
    n = 128
    fs = 48000.0
    freqs = np.fft.rfftfreq(n, 1.0 / fs)

    rng = np.random.default_rng(7)
    H = (rng.standard_normal(len(freqs)) +
         1j * rng.standard_normal(len(freqs))).astype(np.complex128)

    smoothed = smooth_log_magnitude_octave(H, freqs, fraction=12.0)

    assert smoothed[0].imag == 0.0
    assert smoothed[-1].imag == 0.0


def test_limit_complex_magnitude_basic():
    H = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.complex128)
    limited = limit_complex_magnitude(H, max_gain_db=6.0)

    # 6 dB = gain 2.0
    assert np.max(np.abs(limited)) <= 2.0 + 1e-12
    assert np.isclose(limited[0], 0.5)
    assert np.isclose(limited[1], 1.0)


def test_limit_complex_magnitude_preserves_phase():
    H = np.array([2.0 * np.exp(1j * 0.5)], dtype=np.complex128)
    limited = limit_complex_magnitude(H, max_gain_db=0.0)

    assert np.isclose(np.abs(limited[0]), 1.0)
    assert np.isclose(np.angle(limited[0]), 0.5)


def test_limit_complex_magnitude_none():
    H = np.array([1.0, 2.0, 4.0], dtype=np.complex128)
    result = limit_complex_magnitude(H, max_gain_db=None)

    assert np.allclose(result, H)
