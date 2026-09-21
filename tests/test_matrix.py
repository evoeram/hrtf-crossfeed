"""Тесты matrix: Mid = delayed identity, симметрия, прочность."""

import numpy as np

from hrtf_crossfeed.matrix import build_midsafe_matrix_from_side


def _identity_R(n_fft):
    """R = 1+0j на всех частотах — identity (no crossfeed)."""
    return np.ones(n_fft // 2 + 1, dtype=np.complex128)


def _zero_R(n_fft):
    """R = 0+0j на всех частотах — full crossfeed (Mid=0)."""
    return np.zeros(n_fft // 2 + 1, dtype=np.complex128)


def test_mid_is_delayed_identity_strength0():
    """При strength=0 Mid должен быть точно delayed identity."""
    n_fft = 1024
    fs = 48000.0
    ir_len = 256
    latency = 64

    R = _identity_R(n_fft)
    filters, meta, diag = build_midsafe_matrix_from_side(
        R=R, fs=fs, n_fft=n_fft, strength=0.0,
        ir_len=ir_len, latency_samples=latency,
    )

    LL, LR, RL, RR = filters

    # A = delta - B, B = 0 при R=1, strength=0
    assert np.allclose(LR, 0.0, atol=1e-15)
    assert np.allclose(RL, 0.0, atol=1e-15)

    # LL = RR = delta
    assert LL[latency] == 1.0
    assert RR[latency] == 1.0
    assert np.sum(np.abs(LL)) == 1.0

    # Mid error должен быть ~0
    assert meta["mid_error_max_linear"] < 1e-12


def test_mid_is_delayed_identity_strength1_identity_R():
    """При R=identity, strength=1 — Mid всё ещё delayed identity точно."""
    n_fft = 1024
    fs = 48000.0
    ir_len = 256
    latency = 32

    R = _identity_R(n_fft)
    filters, meta, diag = build_midsafe_matrix_from_side(
        R=R, fs=fs, n_fft=n_fft, strength=1.0,
        ir_len=ir_len, latency_samples=latency,
    )

    LL, LR, RL, RR = filters

    # R=1 → B_target = 0 → B = 0 → A = delta
    assert meta["mid_error_max_linear"] < 1e-12
    assert LL[latency] == 1.0


def test_mid_safe_after_truncation():
    """Mid = A + B должен быть delayed identity даже после обрезки FIR.

    Это ключевое свойство mid-safe конструкции: A = delta - B,
    поэтому Mid = delta точно, независимо от B.
    """
    n_fft = 2048
    fs = 48000.0
    ir_len = 128
    latency = 32

    # Произвольный комплексный R (не identity)
    rng = np.random.default_rng(42)
    R = 0.5 + 0.3 * rng.standard_normal(n_fft // 2 + 1) + \
        0.3j * rng.standard_normal(n_fft // 2 + 1)
    R = R.astype(np.complex128)

    filters, meta, diag = build_midsafe_matrix_from_side(
        R=R, fs=fs, n_fft=n_fft, strength=0.7,
        ir_len=ir_len, latency_samples=latency,
    )

    LL, LR, RL, RR = filters

    # Mid = LL + LR (или RL + RR) = delta
    mid = LL + LR
    delta = np.zeros(ir_len, dtype=np.float64)
    delta[latency] = 1.0

    assert np.allclose(mid, delta, atol=1e-10), \
        f"Mid error max = {np.max(np.abs(mid - delta))}"

    # Симметрия: LL=RR, LR=RL
    assert np.allclose(LL, RR, atol=1e-15)
    assert np.allclose(LR, RL, atol=1e-15)


def test_strength_scaling():
    """R_strength = 1 + strength*(R-1): strength=0 → identity, strength=1 → R."""
    n_fft = 512
    fs = 48000.0
    ir_len = 128
    latency = 32

    R = np.full(n_fft // 2 + 1, 0.5 + 0.0j, dtype=np.complex128)

    _, meta_s0, diag_s0 = build_midsafe_matrix_from_side(
        R=R, fs=fs, n_fft=n_fft, strength=0.0,
        ir_len=ir_len, latency_samples=latency,
    )
    _, meta_s1, diag_s1 = build_midsafe_matrix_from_side(
        R=R, fs=fs, n_fft=n_fft, strength=1.0,
        ir_len=ir_len, latency_samples=latency,
    )

    # При strength=0 R_strength = 1 (identity)
    rs0 = diag_s0["R_strength"]
    assert np.allclose(rs0, 1.0 + 0.0j)

    # При strength=1 R_strength = R
    rs1 = diag_s1["R_strength"]
    assert np.allclose(rs1, R)


def test_negative_latency_raises():
    n_fft = 256
    R = _identity_R(n_fft)

    try:
        build_midsafe_matrix_from_side(
            R=R, fs=48000.0, n_fft=n_fft, strength=1.0,
            ir_len=64, latency_samples=-1,
        )
        assert False, "Should have raised"
    except ValueError:
        pass


def test_latency_ge_ir_len_raises():
    n_fft = 256
    R = _identity_R(n_fft)

    try:
        build_midsafe_matrix_from_side(
            R=R, fs=48000.0, n_fft=n_fft, strength=1.0,
            ir_len=64, latency_samples=64,
        )
        assert False, "Should have raised"
    except ValueError:
        pass


def test_short_ir_len_raises():
    n_fft = 256
    R = _identity_R(n_fft)

    try:
        build_midsafe_matrix_from_side(
            R=R, fs=48000.0, n_fft=n_fft, strength=1.0,
            ir_len=4, latency_samples=2,
        )
        assert False, "Should have raised"
    except ValueError:
        pass


def test_meta_fields_present():
    """Проверка наличия всех ключевых полей метаданных."""
    n_fft = 512
    R = _identity_R(n_fft)

    _, meta, _ = build_midsafe_matrix_from_side(
        R=R, fs=48000.0, n_fft=n_fft, strength=1.0,
        ir_len=128, latency_samples=32,
    )

    expected_fields = [
        "strength", "latency_samples", "latency_ms",
        "omitted_circular_energy_ratio", "omitted_circular_energy_db",
        "mid_error_max_linear", "max_matrix_frequency_gain",
        "max_matrix_frequency_gain_db", "recommended_preamp_db",
        "actual_peak_gain_db",
        "actual_side_gain_min_db", "actual_side_gain_max_db",
    ]

    for field in expected_fields:
        assert field in meta, f"Missing meta field: {field}"
