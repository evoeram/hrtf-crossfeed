"""Smoke-тесты: проверка импорта и базовых утилит."""

import numpy as np

from hrtf_crossfeed.utils import db, undb, wrap_deg, tail_fade, rms, ensure_len
from hrtf_crossfeed.itd import woodworth_itd_seconds, validate_or_replace_itd
from hrtf_crossfeed.io import remap_matrix
from hrtf_crossfeed.smoothing import limit_complex_magnitude, smooth_log_magnitude_octave
from hrtf_crossfeed.pipeline import parse_strengths, strength_token


def test_db_undb_roundtrip():
    """dB → undb должен давать исходное значение."""
    x = np.array([0.1, 1.0, 10.0, 100.0])
    recovered = undb(db(x))
    assert np.allclose(recovered, x, rtol=1e-12), f"Roundtrip failed: {recovered} vs {x}"


def test_wrap_deg():
    assert np.isclose(wrap_deg(190.0), -170.0)
    assert np.isclose(wrap_deg(-190.0), 170.0)
    assert np.isclose(wrap_deg(0.0), 0.0)


def test_tail_fade_zeros_tail():
    x = np.ones(100)
    y = tail_fade(x, fraction=0.15)
    assert y[0] == 1.0
    assert np.isclose(y[-1], 0.0)
    assert np.all(y[:80] == 1.0)


def test_rms():
    x = np.array([1.0, -1.0, 1.0, -1.0])
    assert np.isclose(rms(x), 1.0)


def test_ensure_len():
    x = np.array([1.0, 2.0, 3.0])
    assert len(ensure_len(x, 5)) == 5
    assert len(ensure_len(x, 2)) == 2


def test_woodworth_itd_30deg():
    """При az=30° и radius=8.75cm ITD ≈ 261 us."""
    itd = woodworth_itd_seconds(30.0)
    assert 250e-6 < itd < 270e-6, f"ITD at 30° = {itd*1e6:.1f} us, expected ~261 us"


def test_validate_or_replace_itd_measured_mode():
    left, right, info = validate_or_replace_itd(
        itd_left_s=300e-6,
        itd_right_s=310e-6,
        az_deg=30.0,
        mode="measured",
    )
    assert left == 300e-6
    assert right == 310e-6
    assert info["fallback_used"] is False


def test_validate_or_replace_itd_woodworth_mode():
    left, right, info = validate_or_replace_itd(
        itd_left_s=0.0,
        itd_right_s=0.0,
        az_deg=30.0,
        mode="woodworth",
    )
    assert info["fallback_used"] is True
    assert np.isclose(left, right)


def test_validate_or_replace_itd_auto_invalid():
    """При невалидных измеренных ITD режим auto должен переключиться на Woodworth."""
    left, right, info = validate_or_replace_itd(
        itd_left_s=0.0,
        itd_right_s=0.0,
        az_deg=30.0,
        mode="auto",
    )
    assert info["fallback_used"] is True
    assert np.isclose(left, right)
    assert np.isclose(left, woodworth_itd_seconds(30.0))


def test_remap_normal():
    LL, LR, RL, RR = 1, 2, 3, 4
    result = remap_matrix(LL, LR, RL, RR, "normal")
    assert result == [1, 2, 3, 4]


def test_remap_swap_inputs():
    result = remap_matrix(1, 2, 3, 4, "swap_inputs")
    assert result == [2, 1, 4, 3]


def test_remap_swap_outputs():
    result = remap_matrix(1, 2, 3, 4, "swap_outputs")
    assert result == [3, 4, 1, 2]


def test_remap_swap_both():
    result = remap_matrix(1, 2, 3, 4, "swap_both")
    assert result == [4, 3, 2, 1]


def test_limit_complex_magnitude():
    H = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.complex128)
    limited = limit_complex_magnitude(H, max_gain_db=6.0)
    # 6 dB = gain 2.0; всё что выше должно быть ограничено
    assert np.max(np.abs(limited)) <= 2.0 + 1e-12
    # Меньшие значения не должны измениться
    assert np.isclose(limited[0], 0.5)
    assert np.isclose(limited[1], 1.0)


def test_smooth_log_magnitude_octave_disabled():
    H = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    freqs = np.array([0.0, 100.0, 200.0])
    result = smooth_log_magnitude_octave(H, freqs, fraction=0.0)
    assert np.allclose(result, H)


def test_parse_strengths():
    assert parse_strengths("0.70,1.00,1.20") == [0.70, 1.00, 1.20]
    assert parse_strengths("1.0") == [1.0]


def test_strength_token():
    assert strength_token(0.70) == "0p70"
    assert strength_token(1.00) == "1p00"
    assert strength_token(-0.5) == "m0p50"
