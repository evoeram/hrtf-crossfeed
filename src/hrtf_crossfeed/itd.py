"""Модель ITD (Interaural Time Difference) и её валидация.

Содержит единственную (недублированную) реализацию woodworth_itd_seconds
и validate_or_replace_itd. В исходном монолитном файле эти функции
были определены дважды (строки 626-735 и 737-873); вторая версия
перекрывала первую. Здесь сохранена вторая (финальная) версия.
"""

import warnings

import numpy as np


def woodworth_itd_seconds(
    az_deg,
    head_radius_m=0.0875,
    speed_of_sound=343.0,
):
    """
    Приближённый ITD сферической головы для |az| <= 90°.

    При az=30° и radius=8.75 cm получается примерно 261 us.

    Parameters
    ----------
    az_deg : float
        Абсолютный азимут в градусах.
    head_radius_m : float
        Радиус головы в метрах (по умолчанию 8.75 cm).
    speed_of_sound : float
        Скорость звука в м/с.

    Returns
    -------
    float
        ITD в секундах.
    """

    theta = np.deg2rad(abs(float(az_deg)))
    theta = min(theta, np.pi / 2.0)

    return (
        head_radius_m / speed_of_sound
        * (np.sin(theta) + theta)
    )


def validate_or_replace_itd(
    itd_left_s,
    itd_right_s,
    az_deg,
    mode="auto",
):
    """
    Проверяет измеренные ITD и при необходимости заменяет на модель Woodworth.

    Parameters
    ----------
    itd_left_s : float
        Измеренный ITD для левой колонки (секунды).
    itd_right_s : float
        Измеренный ITD для правой колонки (секунды).
    az_deg : float
        Азимут колонок в градусах (для модели Woodworth).
    mode : str
        "measured"  — использовать измеренные значения без проверки;
        "woodworth" — всегда использовать симметричную модель Woodworth;
        "auto"      — использовать измеренные ITD, только если оба значения
                      положительны, правдоподобны и достаточно близки друг к другу.
                      Иначе использовать Woodworth.

    Returns
    -------
    applied_left_s : float
    applied_right_s : float
    result : dict
        Диагностическая информация.
    """

    measured_left = float(itd_left_s)
    measured_right = float(itd_right_s)

    woodworth = float(
        woodworth_itd_seconds(az_deg)
    )

    result = {
        "mode": str(mode),
        "measured_left_seconds": measured_left,
        "measured_right_seconds": measured_right,
        "woodworth_seconds": woodworth,
        "fallback_used": False,
        "fallback_reason": None,
    }

    if mode == "measured":
        applied_left = measured_left
        applied_right = measured_right

    elif mode == "woodworth":
        applied_left = woodworth
        applied_right = woodworth

        result["fallback_used"] = True
        result["fallback_reason"] = "forced Woodworth mode"

    elif mode == "auto":
        min_valid = 20e-6
        max_valid = 900e-6

        left_valid = (
            min_valid <= measured_left <= max_valid
        )
        right_valid = (
            min_valid <= measured_right <= max_valid
        )

        pair_difference = abs(
            measured_left - measured_right
        )
        pair_consistent = pair_difference <= 250e-6

        if left_valid and right_valid and pair_consistent:
            applied_left = measured_left
            applied_right = measured_right
        else:
            reasons = []

            if not left_valid:
                reasons.append(
                    "left measured ITD is invalid: "
                    f"{measured_left * 1e6:.1f} us"
                )

            if not right_valid:
                reasons.append(
                    "right measured ITD is invalid: "
                    f"{measured_right * 1e6:.1f} us"
                )

            if not pair_consistent:
                reasons.append(
                    "left/right ITD difference is too large: "
                    f"{pair_difference * 1e6:.1f} us"
                )

            applied_left = woodworth
            applied_right = woodworth

            result["fallback_used"] = True
            result["fallback_reason"] = "; ".join(reasons)

            warnings.warn(
                "Measured ITD failed validation; using symmetric "
                f"Woodworth ITD {woodworth * 1e6:.1f} us. "
                + result["fallback_reason"]
            )

    else:
        raise ValueError(
            f"Unknown ITD mode: {mode}"
        )

    result["applied_left_seconds"] = float(
        applied_left
    )
    result["applied_right_seconds"] = float(
        applied_right
    )

    return (
        float(applied_left),
        float(applied_right),
        result,
    )
