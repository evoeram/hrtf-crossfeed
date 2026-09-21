"""Тесты pipeline: parse_strengths, strength_token, end-to-end на синтетическом SOFA."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from hrtf_crossfeed.pipeline import (
    parse_strengths,
    run_pipeline,
    strength_token,
)

# ── parse_strengths ──────────────────────────────────────────────


def test_parse_strengths_basic():
    assert parse_strengths("0.70,1.00,1.20") == [0.70, 1.00, 1.20]


def test_parse_strengths_single():
    assert parse_strengths("1.0") == [1.0]


def test_parse_strengths_spaces():
    assert parse_strengths(" 0.5 , 1.0 , 1.5 ") == [0.5, 1.0, 1.5]


def test_parse_strengths_empty_tokens():
    """Пустые токены между запятыми пропускаются."""
    assert parse_strengths("1.0,,2.0") == [1.0, 2.0]


def test_parse_strengths_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_strengths("")


def test_parse_strengths_only_commas_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_strengths(",,,")


def test_parse_strengths_nan_raises():
    with pytest.raises(ValueError, match="not finite"):
        parse_strengths("nan")


def test_parse_strengths_inf_raises():
    with pytest.raises(ValueError, match="not finite"):
        parse_strengths("inf")


# ── strength_token ───────────────────────────────────────────────


def test_strength_token_basic():
    assert strength_token(0.70) == "0p70"
    assert strength_token(1.00) == "1p00"


def test_strength_token_negative():
    assert strength_token(-0.5) == "m0p50"


# ── Синтетический SOFA ───────────────────────────────────────────


@pytest.fixture
def synthetic_sofa(tmp_path):
    """Создаёт минимальный SOFA-файл с синтетическими HRIR."""
    from netCDF4 import Dataset

    sofa_path = tmp_path / "test.sofa"

    fs = 48000.0
    n_samples = 128
    n_measurements = 12

    # Азимуты от -165 до +165 с шагом 30
    azimuths = np.linspace(-165, 165, n_measurements)
    src = np.column_stack([
        azimuths,
        np.zeros(n_measurements),  # elevation = 0
        np.ones(n_measurements),   # radius = 1
    ])

    # Синтетические HRIR: delayed delta для каждого уха
    ir = np.zeros((n_measurements, 2, n_samples), dtype=np.float64)

    for i, az_deg in enumerate(azimuths):
        # Ipsilateral ear: сильный пик
        onset = 10 + int(abs(az_deg) / 30)
        ir[i, 0, onset] = 1.0
        ir[i, 0, onset + 1] = 0.3

        # Contralateral ear: слабее и позже
        ir[i, 1, onset + 5] = 0.5
        ir[i, 1, onset + 6] = 0.2

    with Dataset(str(sofa_path), "w", format="NETCDF4") as ds:
        ds.createDimension("M", n_measurements)
        ds.createDimension("R", 2)
        ds.createDimension("N", n_samples)
        ds.createDimension("I", 3)  # для SourcePosition (az, el, radius)

        ds.SOFAConventions = "SimpleFreeFieldHRIR"
        ds.SOFAConventionsVersion = "1.0"
        ds.DataType = "FIR"
        ds.RoomType = "free field"
        ds.DatabaseName = "synthetic"
        ds.ListenerShortName = "test"

        ir_var = ds.createVariable("Data.IR", "f8", ("M", "R", "N"))
        ir_var[:] = ir

        sr_var = ds.createVariable("Data.SamplingRate", "f8")
        sr_var[:] = fs
        sr_var.Units = "hertz"

        src_var = ds.createVariable("SourcePosition", "f8", ("M", "I"))
        src_var[:] = src
        src_var.Type = "spherical"
        src_var.Units = "degree, degree, metre"

    return str(sofa_path)


def test_run_pipeline_end_to_end(synthetic_sofa, tmp_path):
    """End-to-end пайплайн на синтетическом SOFA."""
    out_prefix = str(tmp_path / "out" / "test")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = run_pipeline(
            sofa_path=synthetic_sofa,
            out_prefix=out_prefix,
            az=30.0,
            ir_len=256,
            latency_samples=64,
            strengths=[0.5, 1.0],
            no_csv=True,
            verbose=False,
        )

    # Метаданные
    assert meta["generator"].startswith("complex HRTF-derived")
    assert meta["fs"] == 48000.0
    assert meta["channel_order"] == ["LL", "LR", "RL", "RR"]
    assert len(meta["outputs"]) == 2

    # Выходные файлы
    for entry in meta["outputs"]:
        wav_path = Path(entry["wav_file"])
        assert wav_path.exists()
        assert wav_path.stat().st_size > 0

        # Проверка mid_error
        assert entry["mid_error_max_linear"] < 1e-10, \
            f"Mid error too large: {entry['mid_error_max_linear']}"

        # Проверка наличия actual_peak_gain_db
        assert "actual_peak_gain_db" in entry


def test_run_pipeline_csv(synthetic_sofa, tmp_path):
    """Пайплайн с CSV — проверка создания CSV и meta.json."""
    out_prefix = str(tmp_path / "out" / "csv_test")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = run_pipeline(
            sofa_path=synthetic_sofa,
            out_prefix=out_prefix,
            az=30.0,
            ir_len=128,
            latency_samples=32,
            strengths=[1.0],
            verbose=False,
        )

    # meta.json
    meta_path = Path(f"{out_prefix}_df_midsafe_meta.json")
    assert meta_path.exists()

    with open(meta_path, encoding="utf-8") as f:
        saved_meta = json.load(f)
    assert saved_meta["fs"] == 48000.0

    # CSV
    for entry in meta["outputs"]:
        csv_path = Path(entry["response_csv"])
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0


def test_run_pipeline_invalid_az(synthetic_sofa, tmp_path):
    with pytest.raises(ValueError, match="az"):
        run_pipeline(
            sofa_path=synthetic_sofa,
            out_prefix=str(tmp_path / "out" / "bad"),
            az=-1.0,
            verbose=False,
        )


def test_run_pipeline_short_ir_len(synthetic_sofa, tmp_path):
    with pytest.raises(ValueError, match="ir-len"):
        run_pipeline(
            sofa_path=synthetic_sofa,
            out_prefix=str(tmp_path / "out" / "bad"),
            ir_len=16,
            verbose=False,
        )


def test_run_pipeline_negative_strength_warns(synthetic_sofa, tmp_path):
    """Отрицательная strength — предупреждение, но не ошибка."""
    out_prefix = str(tmp_path / "out" / "neg")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_pipeline(
            sofa_path=synthetic_sofa,
            out_prefix=out_prefix,
            az=30.0,
            ir_len=128,
            latency_samples=32,
            strengths=[-0.5],
            no_csv=True,
            verbose=False,
        )

    assert any("reverses" in str(wi.message) for wi in w)
