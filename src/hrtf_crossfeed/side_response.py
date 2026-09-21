"""Построение speaker-like Side transfer из HRTF/DTF измерений."""

import numpy as np

from .delay import get_delay_seconds
from .ear_order import determine_ear_map
from .geometry import nearest_source_index, source_positions_to_unit_vectors
from .onset import extract_onset_aligned, onset_index
from .smoothing import limit_complex_magnitude, smooth_log_magnitude_octave
from .sofa import load_sofa
from .itd import validate_or_replace_itd
from .utils import EPS, db


def blend_side_to_identity(
    R,
    freqs,
    start_hz=3500.0,
    end_hz=9000.0,
):
    """
    Плавно переводит complex Side response R к identity R=1
    на высоких частотах.

    Ниже start_hz:
        R_new = R

    Выше end_hz:
        R_new = 1 + 0j

    Между ними используется raised-cosine в log-frequency domain.
    """

    R = np.asarray(R, dtype=np.complex128)
    freqs = np.asarray(freqs, dtype=np.float64)

    if start_hz is None or end_hz is None:
        return R.copy()

    start_hz = float(start_hz)
    end_hz = float(end_hz)

    if start_hz <= 0.0:
        raise ValueError(
            "HF blend start frequency must be positive"
        )

    if end_hz <= start_hz:
        raise ValueError(
            "HF blend end frequency must be greater "
            "than start frequency"
        )

    weight = np.ones_like(freqs)

    above = freqs >= end_hz
    transition = (
        (freqs > start_hz) &
        (freqs < end_hz)
    )

    weight[above] = 0.0

    if np.any(transition):
        log_start = np.log(start_hz)
        log_end = np.log(end_hz)

        x = (
            np.log(freqs[transition]) - log_start
        ) / (
            log_end - log_start
        )

        # 1 -> 0, с нулевой производной на концах.
        weight[transition] = (
            0.5 * (1.0 + np.cos(np.pi * x))
        )

    # Интерполяция от исходного R к identity.
    R_new = 1.0 + weight * (R - 1.0)

    R_new[0] = complex(R_new[0].real, 0.0)

    if len(R_new) > 1:
        R_new[-1] = complex(
            R_new[-1].real,
            0.0,
        )

    return R_new


def build_target_side_response(
    sofa_path,
    az=30.0,
    el=0.0,
    analysis_ir_len=512,
    pre=12,
    n_fft=8192,
    onset_threshold_db=-24.0,
    regularization_db=-40.0,
    smooth_octave=12.0,
    max_side_gain_db=6.0,
    ear_order="auto",
    itd_mode="auto",
    hf_blend_start_hz=3500.0,
    hf_blend_end_hz=9000.0,
):
    """
    Строит целевой complex Side transfer из SOFA-файла.

    Полный пайплайн:
    1. Загрузка SOFA
    2. Поиск ближайших измерений для ±az
    3. Определение порядка ушей
    4. Onset-выравнивание и извлечение HRIR
    5. Восстановление ITD фазовым множителем
    6. Symmetrisation: Hipsi, Hcontra → Hmid, Hside
    7. Регуляризованное complex division: R = Hside/Hmid
    8. Сглаживание magnitude, ограничение усиления, HF blend to identity

    Returns
    -------
    R : np.ndarray, complex
        Целевой Side response.
    fs : float
        Частота дискретизации.
    n_fft : int
        Размер FFT.
    meta : dict
        Метаданные построения.
    analysis : dict
        Промежуточные спектры (freqs, Hipsi, Hcontra, Hmid, Hside, R).
    """

    ir, fs, src, delay_info, sofa_meta = load_sofa(sofa_path)

    m_count, r_count, raw_ir_len = ir.shape

    vectors, src_az, src_el = source_positions_to_unit_vectors(
        src,
        sofa_meta["SourcePositionType"],
        sofa_meta["SourcePositionUnits"],
    )

    # Для стандартной SOFA listener-centric системы:
    # +az = слева, -az = справа.
    idx_left, err_left = nearest_source_index(
        vectors, +az, el
    )
    idx_right, err_right = nearest_source_index(
        vectors, -az, el
    )

    ear_map, ear_info = determine_ear_map(
        ir,
        idx_left,
        idx_right,
        fs,
        ear_order,
    )

    raw_left_ear = ear_map[0]
    raw_right_ear = ear_map[1]

    # Матрица:
    # output ear / input speaker
    #
    # LL: left speaker  -> left ear, ipsilateral
    # RL: left speaker  -> right ear, contralateral
    # LR: right speaker -> left ear, contralateral
    # RR: right speaker -> right ear, ipsilateral

    raw_LL = ir[idx_left, raw_left_ear].copy()
    raw_RL = ir[idx_left, raw_right_ear].copy()
    raw_LR = ir[idx_right, raw_left_ear].copy()
    raw_RR = ir[idx_right, raw_right_ear].copy()

    oLL = onset_index(raw_LL, onset_threshold_db)
    oRL = onset_index(raw_RL, onset_threshold_db)
    oLR = onset_index(raw_LR, onset_threshold_db)
    oRR = onset_index(raw_RR, onset_threshold_db)

    dLL = get_delay_seconds(
        delay_info, idx_left, raw_left_ear, m_count, r_count, fs,
    )
    dRL = get_delay_seconds(
        delay_info, idx_left, raw_right_ear, m_count, r_count, fs,
    )
    dLR = get_delay_seconds(
        delay_info, idx_right, raw_left_ear, m_count, r_count, fs,
    )
    dRR = get_delay_seconds(
        delay_info, idx_right, raw_right_ear, m_count, r_count, fs,
    )

    # Фактическое время прихода = onset Data.IR + Data.Delay.
    tLL = oLL / fs + dLL
    tRL = oRL / fs + dRL
    tLR = oLR / fs + dLR
    tRR = oRR / fs + dRR

    # Относительный contralateral delay отдельно для каждой колонки.
    measured_itd_left_s = tRL - tLL
    measured_itd_right_s = tLR - tRR

    # Выбираем фактически применяемый ITD.
    itd_left_s, itd_right_s, itd_validation = (
        validate_or_replace_itd(
            itd_left_s=measured_itd_left_s,
            itd_right_s=measured_itd_right_s,
            az_deg=az,
            mode=itd_mode,
        )
    )

    hLL = extract_onset_aligned(
        raw_LL, oLL, pre, analysis_ir_len
    )
    hRL = extract_onset_aligned(
        raw_RL, oRL, pre, analysis_ir_len
    )
    hLR = extract_onset_aligned(
        raw_LR, oLR, pre, analysis_ir_len
    )
    hRR = extract_onset_aligned(
        raw_RR, oRR, pre, analysis_ir_len
    )

    n_fft = int(n_fft)

    if n_fft < 2 * analysis_ir_len:
        n_fft = 2 * analysis_ir_len

    if n_fft % 2 != 0:
        n_fft += 1

    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)

    HLL = np.fft.rfft(hLL, n=n_fft)
    HRL = np.fft.rfft(hRL, n=n_fft)
    HLR = np.fft.rfft(hLR, n=n_fft)
    HRR = np.fft.rfft(hRR, n=n_fft)

    # Возвращаем удалённый при onset alignment относительный ITD.
    HRL *= np.exp(-1j * 2.0 * np.pi * freqs * itd_left_s)
    HLR *= np.exp(-1j * 2.0 * np.pi * freqs * itd_right_s)

    # Симметризованная персональная модель.
    Hipsi = 0.5 * (HLL + HRR)
    Hcontra = 0.5 * (HRL + HLR)

    Hmid = Hipsi + Hcontra
    Hside = Hipsi - Hcontra

    max_hmid_power = float(np.max(np.abs(Hmid) ** 2) + EPS)
    reg_power = max_hmid_power * 10.0 ** (
        float(regularization_db) / 10.0
    )

    # Регуляризованное complex division Hside / Hmid.
    R = (
        Hside * np.conj(Hmid) /
        (np.abs(Hmid) ** 2 + reg_power)
    )

    # Сглаживается только magnitude R; phase/ITD остаются исходными.
    R = smooth_log_magnitude_octave(
        R,
        freqs,
        fraction=smooth_octave,
    )

    R_before_limit_max_db = float(np.max(db(R)))

    # Сначала ограничиваем magnitude speaker-like Side target.
    R = limit_complex_magnitude(
        R,
        max_gain_db=max_side_gain_db,
    )

    # Затем плавно убираем высокочастотный crossfeed,
    # переводя комплексный R к identity R=1+0j.
    R = blend_side_to_identity(
        R,
        freqs,
        start_hz=hf_blend_start_hz,
        end_hz=hf_blend_end_hz,
    )

    R[0] = complex(R[0].real, 0.0)
    R[-1] = complex(R[-1].real, 0.0)

    meta = {
        "fs": float(fs),
        "raw_ir_len": int(raw_ir_len),
        "analysis_ir_len": int(analysis_ir_len),
        "n_fft": int(n_fft),
        "sofa_metadata": sofa_meta,
        "selected_positions": {
            "left": {
                "index": int(idx_left),
                "requested_az_deg": float(+az),
                "requested_el_deg": float(el),
                "actual_az_deg": float(src_az[idx_left]),
                "actual_el_deg": float(src_el[idx_left]),
                "angular_error_deg": float(err_left),
                "raw_position": src[idx_left].tolist(),
            },
            "right": {
                "index": int(idx_right),
                "requested_az_deg": float(-az),
                "requested_el_deg": float(el),
                "actual_az_deg": float(src_az[idx_right]),
                "actual_el_deg": float(src_el[idx_right]),
                "angular_error_deg": float(err_right),
                "raw_position": src[idx_right].tolist(),
            },
        },
        "ear_order": {
            "requested": ear_order,
            "raw_to_logical_map": {
                "left_ear_raw_index": int(raw_left_ear),
                "right_ear_raw_index": int(raw_right_ear),
            },
            "detection": ear_info,
        },
        "onsets_samples": {
            "LL": int(oLL),
            "RL": int(oRL),
            "LR": int(oLR),
            "RR": int(oRR),
        },
        "data_delay_seconds": {
            "LL": float(dLL),
            "RL": float(dRL),
            "LR": float(dLR),
            "RR": float(dRR),
        },
        "relative_itd": {
            "measured_left_speaker_seconds": float(
                measured_itd_left_s
            ),
            "measured_right_speaker_seconds": float(
                measured_itd_right_s
            ),
            "measured_left_speaker_samples": float(
                measured_itd_left_s * fs
            ),
            "measured_right_speaker_samples": float(
                measured_itd_right_s * fs
            ),
            "applied_left_speaker_seconds": float(
                itd_left_s
            ),
            "applied_right_speaker_seconds": float(
                itd_right_s
            ),
            "applied_left_speaker_samples": float(
                itd_left_s * fs
            ),
            "applied_right_speaker_samples": float(
                itd_right_s * fs
            ),
            "validation": itd_validation,
        },
        "regularization_db": float(regularization_db),
        "regularization_power": float(reg_power),
        "smooth_octave_fraction": float(smooth_octave),
        "max_side_gain_db": float(max_side_gain_db),
        "hf_identity_blend": {
            "start_hz": float(hf_blend_start_hz),
            "end_hz": float(hf_blend_end_hz),
            "target_above_end_hz": "R = 1 + 0j",
        },
        "side_response_before_limit_max_db": (
            R_before_limit_max_db
        ),
        "side_response_after_limit_max_db": float(
            np.max(db(R))
        ),
    }

    analysis = {
        "freqs": freqs,
        "Hipsi": Hipsi,
        "Hcontra": Hcontra,
        "Hmid": Hmid,
        "Hside": Hside,
        "R": R,
    }

    return R, fs, n_fft, meta, analysis
