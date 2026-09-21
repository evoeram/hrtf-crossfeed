"""Определение порядка ушей (left/right) в Data.IR."""

import warnings

import numpy as np
from scipy import signal

from .utils import EPS


def band_energy(h, fs, f1=1500.0, f2=8000.0, n_fft=4096):
    """
    Средняя энергия импульсной характеристики в полосе частот [f1, f2].
    Используется для определения ipsilateral/contralateral ушей.
    """

    h = np.asarray(h, dtype=np.float64)

    n_fft = max(n_fft, signal.fftconvolve([1.0], [1.0]).size)
    n_fft = max(n_fft, len(h))

    H = np.fft.rfft(h, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)

    mask = (freqs >= f1) & (freqs <= min(f2, fs * 0.49))

    if not np.any(mask):
        return float(np.sum(h * h) + EPS)

    return float(np.mean(np.abs(H[mask]) ** 2) + EPS)


def determine_ear_map(ir, idx_left, idx_right, fs, mode):
    """
    Возвращает raw indices для логических [left_ear, right_ear].

    Parameters
    ----------
    ir : np.ndarray, shape [M, R, N]
        Импульсные характеристики.
    idx_left : int
        Индекс измерения для источника слева.
    idx_right : int
        Индекс измерения для источника справа.
    fs : float
        Частота дискретизации.
    mode : str
        Режим определения: "normal", "swapped" или "auto".

    Returns
    -------
    ear_map : list[int, int]
        [raw_left_ear_idx, raw_right_ear_idx].
    info : dict
        Диагностическая информация о принятом решении.
    """

    if mode == "normal":
        return [0, 1], {"decision": "forced normal"}

    if mode == "swapped":
        return [1, 0], {"decision": "forced swapped"}

    # Для источника слева raw ear 0 должен быть сильнее raw ear 1.
    left_e0 = band_energy(ir[idx_left, 0], fs)
    left_e1 = band_energy(ir[idx_left, 1], fs)

    # Для источника справа raw ear 1 должен быть сильнее raw ear 0.
    right_e1 = band_energy(ir[idx_right, 1], fs)
    right_e0 = band_energy(ir[idx_right, 0], fs)

    normal_score_db = 10.0 * np.log10(
        (left_e0 * right_e1 + EPS) /
        (left_e1 * right_e0 + EPS)
    )

    info = {
        "normal_score_db": float(normal_score_db),
        "decision": None,
    }

    if normal_score_db < -3.0:
        info["decision"] = "auto swapped"
        return [1, 0], info

    if abs(normal_score_db) < 3.0:
        warnings.warn(
            "Ear-order auto detection is inconclusive "
            f"({normal_score_db:+.2f} dB). Keeping normal order."
        )
        info["decision"] = "auto inconclusive, kept normal"
        return [0, 1], info

    info["decision"] = "auto normal"
    return [0, 1], info
