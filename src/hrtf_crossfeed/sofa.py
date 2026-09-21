"""Чтение HRTF/DTF данных из SOFA (netCDF4) файлов."""

import warnings
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from .utils import safe_attr, validate_finite


def load_sofa(path):
    """
    Загружает HRTF/DTF измерения из SOFA-файла.

    Parameters
    ----------
    path : str | Path
        Путь к SOFA-файлу.

    Returns
    -------
    ir : np.ndarray, shape [M, R=2, N]
        Импульсные характеристики для каждого измерения и уха.
    fs : float
        Частота дискретизации (одинаковая для всех измерений).
    src : np.ndarray, shape [M, >=2]
        Позиции источников.
    delay_info : dict | None
        Данные о задержках Data.Delay, либо None.
    metadata : dict
        Метаданные SOFA-файла.
    """

    path = Path(path)

    with Dataset(str(path), "r") as ds:
        if "Data.IR" not in ds.variables:
            raise ValueError("SOFA file has no Data.IR variable")

        if "Data.SamplingRate" not in ds.variables:
            raise ValueError("SOFA file has no Data.SamplingRate variable")

        if "SourcePosition" not in ds.variables:
            if "ListenerPosition" in ds.variables:
                warnings.warn(
                    "SOFA file has no SourcePosition; "
                    "falling back to ListenerPosition"
                )
                src_var = ds.variables["ListenerPosition"]
            else:
                raise ValueError(
                    "SOFA file has no SourcePosition or ListenerPosition variable"
                )
        else:
            src_var = ds.variables["SourcePosition"]

        ir_var = ds.variables["Data.IR"]
        sr_var = ds.variables["Data.SamplingRate"]

        ir = np.asarray(ir_var[:], dtype=np.float64)
        sr_all = np.asarray(sr_var[:], dtype=np.float64).reshape(-1)
        src = np.asarray(src_var[:], dtype=np.float64)

        validate_finite("Data.IR", ir)
        validate_finite("Data.SamplingRate", sr_all)
        validate_finite("SourcePosition", src)

        if ir.ndim != 3:
            raise ValueError(
                f"Expected Data.IR shape [M,R,N], got {ir.shape}"
            )

        if ir.shape[1] != 2:
            raise ValueError(
                f"Expected two receivers/ears, got Data.IR shape {ir.shape}"
            )

        if len(sr_all) == 0:
            raise ValueError("Empty Data.SamplingRate")

        if np.max(np.abs(sr_all - sr_all[0])) > 1e-6:
            raise ValueError(
                "Variable sampling rate across measurements is not supported"
            )

        fs = float(sr_all[0])

        src = np.squeeze(src)

        if src.ndim != 2 or src.shape[1] < 2:
            raise ValueError(
                f"Unexpected SourcePosition shape: {src.shape}"
            )

        delay_info = None

        if "Data.Delay" in ds.variables:
            delay_var = ds.variables["Data.Delay"]

            delay_info = {
                "data": np.asarray(delay_var[:], dtype=np.float64),
                "dimensions": list(delay_var.dimensions),
                "units": safe_attr(delay_var, "Units", ""),
            }

        metadata = {
            "SOFAConventions": safe_attr(ds, "SOFAConventions", ""),
            "SOFAConventionsVersion": safe_attr(
                ds, "SOFAConventionsVersion", ""
            ),
            "DataType": safe_attr(ds, "DataType", ""),
            "RoomType": safe_attr(ds, "RoomType", ""),
            "DatabaseName": safe_attr(ds, "DatabaseName", ""),
            "ListenerShortName": safe_attr(ds, "ListenerShortName", ""),
            "SourcePositionType": safe_attr(src_var, "Type", "spherical"),
            "SourcePositionUnits": safe_attr(
                src_var, "Units", "degree, degree, metre"
            ),
            "DataDelayUnits": (
                delay_info["units"] if delay_info is not None else None
            ),
        }

    return ir, fs, src, delay_info, metadata
