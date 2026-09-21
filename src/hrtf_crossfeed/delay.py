"""Обработка Data.Delay из SOFA-файлов."""

import warnings

import numpy as np


def delay_value(delay_info, measurement_idx, raw_ear_idx, m_count, r_count):
    """
    Извлекает значение Data.Delay, используя имена SOFA dimensions,
    а при их отсутствии — размеры массива.
    """

    if delay_info is None:
        return 0.0

    d = np.asarray(delay_info["data"], dtype=np.float64)
    dims = list(delay_info.get("dimensions", []))

    if d.ndim == 0:
        return float(d)

    if len(dims) != d.ndim:
        dims = [""] * d.ndim

    selection = []

    for axis, (dim_name, axis_size) in enumerate(zip(dims, d.shape)):
        dim_upper = str(dim_name).upper()

        if dim_upper == "M":
            selection.append(min(measurement_idx, axis_size - 1))
            continue

        if dim_upper == "R":
            selection.append(min(raw_ear_idx, axis_size - 1))
            continue

        if axis_size == 1:
            selection.append(0)
            continue

        # Fallback по размеру.
        if axis_size == m_count and axis_size != r_count:
            selection.append(min(measurement_idx, axis_size - 1))
        elif axis_size == r_count:
            selection.append(min(raw_ear_idx, axis_size - 1))
        elif axis_size == m_count:
            selection.append(min(measurement_idx, axis_size - 1))
        else:
            warnings.warn(
                "Could not confidently interpret Data.Delay shape "
                f"{d.shape}, dimensions={dims}; using index 0 on axis {axis}"
            )
            selection.append(0)

    return float(d[tuple(selection)])


def delay_to_seconds(value, units, fs):
    """Преобразует значение задержки в секунды с учётом единиц измерения."""

    value = float(value)

    # Для нулевой задержки единицы не имеют значения.
    if abs(value) < 1e-15:
        return 0.0

    units_lower = str(units or "").strip().lower()

    if "sample" in units_lower:
        return value / float(fs)

    if (
        "second" in units_lower
        or units_lower in {"s", "sec", "secs"}
        or " sec" in units_lower
    ):
        return value

    if units_lower == "":
        warnings.warn(
            "Non-zero Data.Delay has no Units attribute; "
            "assuming seconds"
        )
        return value

    warnings.warn(
        f"Unknown Data.Delay.Units={units!r}; assuming seconds"
    )
    return value


def get_delay_seconds(
    delay_info,
    measurement_idx,
    raw_ear_idx,
    m_count,
    r_count,
    fs,
):
    """Полное извлечение задержки в секундах для конкретного измерения и уха."""

    if delay_info is None:
        return 0.0

    value = delay_value(
        delay_info,
        measurement_idx,
        raw_ear_idx,
        m_count,
        r_count,
    )

    return delay_to_seconds(
        value,
        delay_info.get("units", ""),
        fs,
    )
