"""Высокоуровневый пайплайн: связывает side_response → matrix → io."""

import json
import logging
import warnings
from pathlib import Path

import numpy as np

from .io import remap_matrix, save_frequency_csv, save_llrr
from .matrix import build_midsafe_matrix_from_side
from .side_response import build_target_side_response

logger = logging.getLogger(__name__)


def parse_strengths(text):
    """Разбирает строку вида "0.70,1.00,1.20" в список float."""

    values = []

    for token in text.split(","):
        token = token.strip()

        if not token:
            continue

        value = float(token)

        if not np.isfinite(value):
            raise ValueError(
                f"Invalid strength value: {token!r} (not finite)"
            )

        values.append(value)

    if not values:
        raise ValueError("Strength list is empty")

    return values


def strength_token(value):
    """Формирует безопасный токен для имени файла из значения strength."""

    text = f"{value:.2f}"
    return text.replace("-", "m").replace(".", "p")


def run_pipeline(
    sofa_path,
    out_prefix,
    az=30.0,
    el=0.0,
    analysis_ir_len=512,
    ir_len=512,
    pre=12,
    n_fft=8192,
    latency_samples=128,
    onset_threshold_db=-24.0,
    regularization_db=-40.0,
    smooth_octave=12.0,
    max_side_gain_db=6.0,
    ear_order="auto",
    itd_mode="auto",
    hf_blend_start_hz=3500.0,
    hf_blend_end_hz=9000.0,
    tail_fraction=0.15,
    strengths=None,
    remap="normal",
    no_csv=False,
    verbose=True,
):
    """
    Полный пайплайн генерации crossfeed матрицы из SOFA-файла.

    Parameters
    ----------
    sofa_path : str | Path
        Путь к входному SOFA-файлу.
    out_prefix : str
        Префикс для выходных файлов (WAV, CSV, JSON).
    az : float
        Азимут колонок в градусах.
    el : float
        Угол возвышения в градусах.
    analysis_ir_len : int
        Длина HRIR для спектрального анализа.
    ir_len : int
        Длина выходных FIR фильтров.
    pre : int
        Сэмплы до onset.
    n_fft : int
        Размер FFT.
    latency_samples : int
        Общая задержка (в сэмплах).
    onset_threshold_db : float
        Порог onset-детекции.
    regularization_db : float
        Регуляризация complex division.
    smooth_octave : float
        Сглаживание magnitude (fraction octave).
    max_side_gain_db : float
        Максимальное усиление Side response.
    ear_order : str
        Режим определения ушей: "auto", "normal", "swapped".
    itd_mode : str
        Режим ITD: "auto", "measured", "woodworth".
    hf_blend_start_hz : float
        Начало HF blend to identity.
    hf_blend_end_hz : float
        Конец HF blend to identity.
    tail_fraction : float
        Доля косинусного затухания.
    strengths : list[float] | None
        Список значений strength. По умолчанию [0.70, 1.00, 1.20].
    remap : str
        Режим перестановки каналов.
    no_csv : bool
        Не сохранять CSV файлы.
    verbose : bool
        Печатать прогресс.

    Returns
    -------
    dict
        Полные метаданные со списком сгенерированных файлов.
    """

    if strengths is None:
        strengths = [0.70, 1.00, 1.20]

    if az < 0:
        raise ValueError("--az should be non-negative")

    if analysis_ir_len < 32:
        raise ValueError("--analysis-ir-len is too short")

    if ir_len < 32:
        raise ValueError("--ir-len is too short")

    if pre < 0 or pre >= analysis_ir_len:
        raise ValueError(
            "--pre must satisfy 0 <= pre < analysis-ir-len"
        )

    if not 0.0 <= tail_fraction <= 0.5:
        raise ValueError(
            "--tail-fraction should be between 0 and 0.5"
        )

    for strength in strengths:
        if strength < 0:
            warnings.warn(
                f"Negative strength {strength} reverses the intended "
                "crossfeed direction"
            )

        if strength > 1.0:
            warnings.warn(
                f"Strength {strength} is an extrapolation beyond the "
                "measured speaker target"
            )

    R, fs, n_fft_used, target_meta, analysis = (
        build_target_side_response(
            sofa_path=sofa_path,
            az=az,
            el=el,
            analysis_ir_len=analysis_ir_len,
            pre=pre,
            n_fft=n_fft,
            onset_threshold_db=onset_threshold_db,
            regularization_db=regularization_db,
            smooth_octave=smooth_octave,
            max_side_gain_db=max_side_gain_db,
            ear_order=ear_order,
            itd_mode=itd_mode,
            hf_blend_start_hz=hf_blend_start_hz,
            hf_blend_end_hz=hf_blend_end_hz,
        )
    )

    output_meta = {
        "generator": "complex HRTF-derived mid-safe crossfeed v2",
        "sofa": str(sofa_path),
        "output_prefix": str(out_prefix),
        "fs": float(fs),
        "channel_order": ["LL", "LR", "RL", "RR"],
        "matrix_definition": {
            "rows": ["left ear", "right ear"],
            "columns": ["left input", "right input"],
            "mid_target": "delayed identity",
            "side_target": (
                "(Hipsi-Hcontra)/(Hipsi+Hcontra)"
            ),
        },
        "remap": remap,
        "output_ir_len": int(ir_len),
        "latency_samples": int(latency_samples),
        "analysis": target_meta,
        "outputs": [],
    }

    freqs = analysis["freqs"]

    for strength in strengths:
        hs, matrix_meta, diagnostics = (
            build_midsafe_matrix_from_side(
                R=R,
                fs=fs,
                n_fft=n_fft_used,
                strength=strength,
                ir_len=ir_len,
                latency_samples=latency_samples,
                tail_fraction=tail_fraction,
            )
        )

        hs_remapped = remap_matrix(
            *hs,
            mode=remap,
        )

        token = strength_token(strength)
        base_path = f"{out_prefix}_df_midsafe_s{token}"

        wav_path = f"{base_path}.wav"
        save_llrr(
            wav_path,
            fs,
            *hs_remapped,
        )

        csv_path = None

        if not no_csv:
            csv_path = f"{base_path}_response.csv"

            save_frequency_csv(
                csv_path,
                freqs,
                analysis["Hipsi"],
                analysis["Hcontra"],
                R,
                diagnostics,
            )

        output_entry = {
            **matrix_meta,
            "wav_file": wav_path,
            "response_csv": csv_path,
        }

        output_meta["outputs"].append(output_entry)

        if verbose:
            logger.info("")
            logger.info("Saved: %s", wav_path)

            if csv_path is not None:
                logger.info("CSV:   %s", csv_path)

            logger.info("  strength:                 %.3f", strength)
            logger.info(
                "  latency:                  %.3f ms",
                matrix_meta['latency_ms'],
            )
            logger.info(
                "  omitted FIR energy:       %.1f dB",
                matrix_meta['omitted_circular_energy_db'],
            )
            logger.info(
                "  max matrix gain:          %+.2f dB",
                matrix_meta['max_matrix_frequency_gain_db'],
            )
            logger.info(
                "  recommended preamp:       %+.2f dB",
                matrix_meta['recommended_preamp_db'],
            )
            logger.info(
                "  numerical Mid error:      %.3e",
                matrix_meta['mid_error_max_linear'],
            )

    meta_path = Path(f"{out_prefix}_df_midsafe_meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path.write_text(
        json.dumps(
            output_meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if verbose:
        logger.info("")
        logger.info("Metadata: %s", meta_path)
        logger.info("")
        logger.info("Expected convolution channel order: LL, LR, RL, RR")
        logger.info(
            "All filters contain an intentional common delay of %d samples.",
            latency_samples,
        )

    return output_meta
