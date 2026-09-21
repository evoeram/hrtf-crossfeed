"""Remap каналов и сохранение WAV/CSV."""

from pathlib import Path

import numpy as np
import soundfile as sf

from .utils import db


def remap_matrix(LL, LR, RL, RR, mode):
    """
    Переставляет каналы для convolution plugin.

    Parameters
    ----------
    LL, LR, RL, RR : np.ndarray
        FIR фильтры в исходном порядке.
    mode : str
        "normal"       — без изменений;
        "swap_inputs"  — перестановка входов;
        "swap_outputs" — перестановка выходов;
        "swap_both"    — перестановка и входов, и выходов.

    Returns
    -------
    list[np.ndarray]
        [LL, LR, RL, RR] в новом порядке.
    """

    if mode == "normal":
        return [LL, LR, RL, RR]

    if mode == "swap_inputs":
        return [LR, LL, RR, RL]

    if mode == "swap_outputs":
        return [RL, RR, LL, LR]

    if mode == "swap_both":
        return [RR, RL, LR, LL]

    raise ValueError(f"Unknown remap mode: {mode}")


def save_llrr(path, fs, LL, LR, RL, RR):
    """
    Сохраняет 4-канальный WAV (LL, LR, RL, RR) в float32.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y = np.stack(
        [LL, LR, RL, RR],
        axis=1,
    ).astype(np.float32)

    # Обязательно float WAV, без PCM16-квантования.
    sf.write(
        str(path),
        y,
        int(round(fs)),
        format="WAV",
        subtype="FLOAT",
    )


def save_frequency_csv(
    path,
    freqs,
    Hipsi,
    Hcontra,
    R,
    diagnostics,
):
    """
    Сохраняет частотные отклики в CSV для последующего анализа.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    R_actual = diagnostics["R_actual"]
    Hmid_actual = diagnostics["Hmid_actual"]
    Hside_actual = diagnostics["Hside_actual"]

    data = np.column_stack(
        [
            freqs,
            db(Hipsi),
            db(Hcontra),
            db(R),
            np.unwrap(np.angle(R)),
            db(R_actual),
            np.unwrap(np.angle(R_actual)),
            db(Hmid_actual),
            db(Hside_actual),
        ]
    )

    header = ",".join(
        [
            "frequency_hz",
            "ipsi_db",
            "contra_db",
            "target_side_db",
            "target_side_phase_rad",
            "actual_side_db",
            "actual_side_phase_rad",
            "actual_mid_db",
            "actual_side_raw_db",
        ]
    )

    np.savetxt(
        str(path),
        data,
        delimiter=",",
        header=header,
        comments="",
    )
