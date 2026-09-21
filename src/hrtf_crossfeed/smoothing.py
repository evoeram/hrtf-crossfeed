"""Сглаживание magnitude в octave domain и ограничение усиления."""

import numpy as np

from .utils import undb


def smooth_log_magnitude_octave(H, freqs, fraction=0.0):
    """
    Сглаживает только magnitude, сохраняя исходную complex phase.

    Parameters
    ----------
    H : np.ndarray, complex
        Комплексный частотный отклик.
    freqs : np.ndarray
        Частоты (Гц), соответствующие bins.
    fraction : float
        0  — выключено;
        6  — примерно 1/6 octave FWHM;
        12 — примерно 1/12 octave FWHM.

    Returns
    -------
    np.ndarray, complex
        Сглаженный отклик с сохранённой фазой.
    """

    H = np.asarray(H, dtype=np.complex128)
    freqs = np.asarray(freqs, dtype=np.float64)

    if fraction is None or fraction <= 0:
        return H.copy()

    out = H.copy()
    mag = np.maximum(np.abs(H), 1e-12)
    phase = np.angle(H)
    log_mag = np.log(mag)

    positive = freqs > 0.0

    if np.sum(positive) < 4:
        return H.copy()

    f_pos = freqs[positive]
    x = np.log2(f_pos)
    lm = log_mag[positive]

    # fraction задаёт приблизительную FWHM в октавах.
    fwhm_oct = 1.0 / float(fraction)
    sigma = fwhm_oct / 2.354820045

    # Векторизованное сглаживание: матрица весов Гаусса в log-frequency domain.
    dx = x[:, None] - x[None, :]
    weights = np.exp(-0.5 * (dx / sigma) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)

    smoothed = weights @ lm

    new_mag = mag.copy()
    new_mag[positive] = np.exp(smoothed)

    out = new_mag * np.exp(1j * phase)

    # DC и Nyquist у real FIR должны быть действительными.
    out[0] = complex(out[0].real, 0.0)

    if len(out) > 1:
        out[-1] = complex(out[-1].real, 0.0)

    return out


def limit_complex_magnitude(H, max_gain_db):
    """
    Ограничивает magnitude комплексного отклика заданным уровнем (dB).
    Фаза сохраняется.
    """

    H = np.asarray(H, dtype=np.complex128).copy()

    if max_gain_db is None:
        return H

    max_gain = undb(max_gain_db)
    mag = np.abs(H)

    mask = mag > max_gain

    if np.any(mask):
        H[mask] *= max_gain / mag[mask]

    return H
