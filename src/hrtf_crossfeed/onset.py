"""Onset-детекция и извлечение выровненных HRIR."""

import numpy as np

from .utils import EPS, tail_fade, undb


def onset_index(h, threshold_db=-24.0):
    """
    Energy-envelope onset.

    Parameters
    ----------
    h : np.ndarray
        Импульсная характеристика.
    threshold_db : float
        Порог огибающей относительно максимума (в dB).

    Returns
    -------
    int
        Индекс начала импульсной характеристики.
    """

    h = np.asarray(h, dtype=np.float64)

    if len(h) == 0:
        return 0

    abs_h = np.abs(h)

    if np.max(abs_h) <= EPS:
        return 0

    # Небольшая локальная RMS/envelope для устойчивости к одиночным пикам.
    kernel_len = min(9, len(h))

    if kernel_len % 2 == 0:
        kernel_len -= 1

    kernel_len = max(kernel_len, 1)

    kernel = np.ones(kernel_len, dtype=np.float64) / kernel_len
    env2 = np.convolve(h * h, kernel, mode="same")
    env = np.sqrt(np.maximum(env2, 0.0))

    threshold = np.max(env) * undb(threshold_db)
    candidates = np.flatnonzero(env >= threshold)

    if len(candidates) == 0:
        return int(np.argmax(env))

    return int(candidates[0])


def extract_onset_aligned(
    h,
    onset,
    pre,
    out_len,
    tail_fraction=0.15,
):
    """
    Извлекает HRIR так, чтобы onset оказался точно на индексе pre.

    Это удаляет абсолютное время прихода, которое потом восстанавливается
    отдельно фазовым множителем.
    """

    h = np.asarray(h, dtype=np.float64)
    y = np.zeros(out_len, dtype=np.float64)

    src_start = max(0, onset - pre)
    dst_start = pre - (onset - src_start)

    available = min(
        len(h) - src_start,
        out_len - dst_start,
    )

    if available > 0:
        y[dst_start:dst_start + available] = (
            h[src_start:src_start + available]
        )

    y = tail_fade(y, tail_fraction)

    return y
