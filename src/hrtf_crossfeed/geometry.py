"""Преобразование координат SourcePosition и поиск ближайшего направления."""

import warnings

import numpy as np

from .utils import EPS, wrap_deg


def source_positions_to_unit_vectors(src, position_type, units):
    """
    Преобразует SourcePosition в единичные векторы и углы.

    Parameters
    ----------
    src : np.ndarray, shape [M, >=2]
        Позиции источников (сферические или декартовы).
    position_type : str
        Тип координат: "spherical" или "cartesian".
    units : str
        Единицы измерения для сферических координат.

    Returns
    -------
    vectors : np.ndarray, shape [M, 3]
        Единичные векторы направлений.
    azimuth_deg : np.ndarray, shape [M]
        Азимут в градусах.
    elevation_deg : np.ndarray, shape [M]
        Угол возвышения в градусах.
    """

    src = np.asarray(src, dtype=np.float64)

    position_type = str(position_type).strip().lower()
    units_lower = str(units).strip().lower()

    if "cartesian" in position_type:
        if src.shape[1] < 3:
            raise ValueError(
                "Cartesian SourcePosition requires at least 3 columns"
            )

        xyz = src[:, :3].copy()
        norms = np.linalg.norm(xyz, axis=1)

        if np.any(norms < EPS):
            raise ValueError("SourcePosition contains zero Cartesian vector")

        vectors = xyz / norms[:, None]

        az = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0]))
        el = np.rad2deg(
            np.arctan2(
                vectors[:, 2],
                np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2),
            )
        )

        return vectors, wrap_deg(az), el

    if "spherical" not in position_type and position_type:
        warnings.warn(
            f"Unknown SourcePosition.Type={position_type!r}; "
            "assuming spherical coordinates"
        )

    az = src[:, 0].copy()
    el = src[:, 1].copy()

    if "radian" in units_lower or " rad" in units_lower:
        az = np.rad2deg(az)
        el = np.rad2deg(el)
    elif "degree" not in units_lower and "deg" not in units_lower:
        warnings.warn(
            f"Unknown SourcePosition.Units={units!r}; "
            "assuming azimuth/elevation in degrees"
        )

    az_rad = np.deg2rad(az)
    el_rad = np.deg2rad(el)

    cos_el = np.cos(el_rad)

    vectors = np.column_stack(
        [
            cos_el * np.cos(az_rad),
            cos_el * np.sin(az_rad),
            np.sin(el_rad),
        ]
    )

    return vectors, wrap_deg(az), el


def direction_to_unit_vector(az_deg, el_deg):
    """Единичный вектор для азимута и угла возвышения в градусах."""

    az = np.deg2rad(float(az_deg))
    el = np.deg2rad(float(el_deg))

    return np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=np.float64,
    )


def nearest_source_index(vectors, az_deg, el_deg):
    """
    Индекс ближайшего измерения к заданному направлению.

    Returns
    -------
    idx : int
        Индекс ближайшего измерения.
    angular_error_deg : float
        Угловая ошибка в градусах.
    """

    target = direction_to_unit_vector(az_deg, el_deg)

    dots = np.clip(vectors @ target, -1.0, 1.0)
    angular_errors = np.rad2deg(np.arccos(dots))

    idx = int(np.argmin(angular_errors))

    return idx, float(angular_errors[idx])
