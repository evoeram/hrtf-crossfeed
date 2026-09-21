"""Построение mid-safe FIR матрицы из Side response."""

import numpy as np

from .utils import EPS, db, tail_fade


def build_midsafe_matrix_from_side(
    R,
    fs,
    n_fft,
    strength,
    ir_len,
    latency_samples,
    tail_fraction=0.15,
):
    """
    Строит симметричную 2×2 FIR матрицу crossfeed.

    Target:
        Mid  = delayed identity
        Side = delayed R_strength

    R_strength = 1 + strength * (R - 1)

    Matrix:
        A = (1 + R_strength) / 2
        B = (1 - R_strength) / 2

    Реализация строится через B, после чего:
        A = delayed_delta - B

    Поэтому Mid сохраняется точно даже после обрезки FIR.

    Parameters
    ----------
    R : np.ndarray, complex
        Целевой Side response.
    fs : float
        Частота дискретизации.
    n_fft : int
        Размер FFT.
    strength : float
        Сила crossfeed: 0 = identity, 1 = measured target, >1 = extrapolation.
    ir_len : int
        Длина выходных FIR фильтров.
    latency_samples : int
        Общая задержка direct и cross путей (в сэмплов).
    tail_fraction : float
        Доля FIR, используемая для косинусного затухания.

    Returns
    -------
    filters : list[np.ndarray]
        [LL, LR, RL, RR] — четыре FIR фильтра.
    meta : dict
        Метрик: omitted energy, max gain, recommended preamp, и т.д.
    diagnostics : dict
        R_strength, R_actual, Hmid_actual, Hside_actual.
    """

    R = np.asarray(R, dtype=np.complex128)

    ir_len = int(ir_len)
    latency_samples = int(latency_samples)

    if ir_len < 8:
        raise ValueError("--ir-len must be at least 8")

    if latency_samples < 0 or latency_samples >= ir_len:
        raise ValueError(
            "--latency-samples must satisfy "
            "0 <= latency < ir_len"
        )

    R_strength = 1.0 + float(strength) * (R - 1.0)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)

    common_delay = np.exp(
        -1j * 2.0 * np.pi * freqs *
        (latency_samples / fs)
    )

    # Off-diagonal transfer.
    B_target = 0.5 * (1.0 - R_strength) * common_delay

    B_target[0] = complex(B_target[0].real, 0.0)
    B_target[-1] = complex(B_target[-1].real, 0.0)

    b_circular = np.fft.irfft(B_target, n=n_fft)

    total_energy = float(
        np.sum(b_circular * b_circular) + EPS
    )
    omitted_energy = float(
        np.sum(b_circular[ir_len:] ** 2)
    )

    omitted_ratio = omitted_energy / total_energy

    B = b_circular[:ir_len].copy()
    B = tail_fade(B, tail_fraction)

    delta = np.zeros(ir_len, dtype=np.float64)
    delta[latency_samples] = 1.0

    # A + B = delta точно.
    A = delta - B

    LL = A.copy()
    LR = B.copy()
    RL = B.copy()
    RR = A.copy()

    # Анализ уже фактически получившегося FIR.
    A_actual = np.fft.rfft(A, n=n_fft)
    B_actual = np.fft.rfft(B, n=n_fft)
    D_actual = np.fft.rfft(delta, n=n_fft)

    Hmid_actual = A_actual + B_actual
    Hside_actual = A_actual - B_actual

    # Убираем общую задержку для анализа Side response.
    R_actual = Hside_actual / np.where(
        np.abs(D_actual) > EPS,
        D_actual,
        1.0,
    )

    mid_error = Hmid_actual - D_actual
    mid_error_max = float(np.max(np.abs(mid_error)))

    # Для симметричной матрицы singular values равны
    # |A+B| и |A-B|.
    max_matrix_gain = float(
        max(
            np.max(np.abs(Hmid_actual)),
            np.max(np.abs(Hside_actual)),
        )
    )

    recommended_preamp_db = -max(
        0.0,
        20.0 * np.log10(max(max_matrix_gain, EPS)),
    )

    meta = {
        "strength": float(strength),
        "latency_samples": int(latency_samples),
        "latency_ms": float(
            1000.0 * latency_samples / fs
        ),
        "omitted_circular_energy_ratio": float(
            omitted_ratio
        ),
        "omitted_circular_energy_db": float(
            10.0 * np.log10(max(omitted_ratio, EPS))
        ),
        "mid_error_max_linear": mid_error_max,
        "max_matrix_frequency_gain": max_matrix_gain,
        "max_matrix_frequency_gain_db": float(
            20.0 * np.log10(
                max(max_matrix_gain, EPS)
            )
        ),
        "recommended_preamp_db": float(
            recommended_preamp_db
        ),
        "actual_side_gain_min_db": float(
            np.min(db(R_actual))
        ),
        "actual_side_gain_max_db": float(
            np.max(db(R_actual))
        ),
    }

    diagnostics = {
        "R_strength": R_strength,
        "R_actual": R_actual,
        "Hmid_actual": Hmid_actual,
        "Hside_actual": Hside_actual,
    }

    return [LL, LR, RL, RR], meta, diagnostics
