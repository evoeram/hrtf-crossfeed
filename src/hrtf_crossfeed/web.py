"""Flask web-интерфейс для hrtf-crossfeed."""

import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from .pipeline import parse_strengths, run_pipeline, strength_token
from .sofa import load_sofa

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=None)

# Рабочая директория для загрузок и результатов
WORK_DIR = Path(tempfile.gettempdir()) / "hrtf_crossfeed_web"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Таймаут для сессий (просто очистка старых папок)
MAX_SESSIONS = 20


def _cleanup_old_sessions():
    """Удаляет старые сессии, оставляя не более MAX_SESSIONS."""
    sessions = sorted(WORK_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
    while len(sessions) > MAX_SESSIONS:
        oldest = sessions.pop(0)
        shutil.rmtree(oldest, ignore_errors=True)


@app.route("/")
def index():
    html_path = Path(__file__).parent / "web_static" / "index.html"
    return send_file(html_path, mimetype="text/html")


@app.route("/api/upload", methods=["POST"])
def upload_sofa():
    """Загрузка SOFA-файла, возврат метаданных."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.lower().endswith(".sofa"):
        return jsonify({"error": "File must have .sofa extension"}), 400

    session_id = uuid.uuid4().hex[:12]
    session_dir = WORK_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    sofa_path = session_dir / file.filename
    file.save(str(sofa_path))

    _cleanup_old_sessions()

    try:
        ir, fs, src, delay_info, metadata = load_sofa(str(sofa_path))
    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        return jsonify({"error": f"Failed to load SOFA: {e}"}), 400

    n_measurements, n_ears, n_samples = ir.shape

    # Вычисляем диапазон азимутов
    import numpy as np

    from .geometry import source_positions_to_unit_vectors

    try:
        vectors, az_arr, el_arr = source_positions_to_unit_vectors(
            src, metadata["SourcePositionType"],
            metadata["SourcePositionUnits"],
        )
        az_min = float(np.min(az_arr))
        az_max = float(np.max(az_arr))
        az_list = sorted(set(round(float(a), 1) for a in az_arr))
    except Exception:
        az_min, az_max, az_list = -180.0, 180.0, []

    info = {
        "session_id": session_id,
        "filename": file.filename,
        "fs": float(fs),
        "n_measurements": int(n_measurements),
        "n_ears": int(n_ears),
        "n_samples": int(n_samples),
        "sofa_conventions": metadata.get("SOFAConventions", ""),
        "database_name": metadata.get("DatabaseName", ""),
        "listener": metadata.get("ListenerShortName", ""),
        "data_type": metadata.get("DataType", ""),
        "room_type": metadata.get("RoomType", ""),
        "azimuth_min": az_min,
        "azimuth_max": az_max,
        "azimuths": az_list[:50],
    }

    return jsonify(info)


@app.route("/api/generate", methods=["POST"])
def generate():
    """Запуск пайплайна с параметрами."""
    data = request.get_json(force=True)

    session_id = data.get("session_id", "")
    session_dir = WORK_DIR / session_id

    if not session_dir.is_dir():
        return jsonify({"error": "Invalid or expired session"}), 400

    # Найти SOFA файл в директории сессии
    sofa_files = list(session_dir.glob("*.sofa"))
    if not sofa_files:
        return jsonify({"error": "SOFA file not found"}), 400

    sofa_path = sofa_files[0]

    # Параметры
    out_prefix = str(session_dir / "out" / "result")

    try:
        strengths_text = data.get("strengths", "0.70,1.00,1.20")
        strengths = parse_strengths(strengths_text)

        import warnings
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            meta = run_pipeline(
                sofa_path=str(sofa_path),
                out_prefix=out_prefix,
                az=float(data.get("az", 30.0)),
                el=float(data.get("el", 0.0)),
                analysis_ir_len=int(data.get("analysis_ir_len", 512)),
                ir_len=int(data.get("ir_len", 512)),
                pre=int(data.get("pre", 12)),
                n_fft=int(data.get("n_fft", 8192)),
                latency_samples=int(data.get("latency_samples", 128)),
                onset_threshold_db=float(data.get("onset_threshold_db", -24.0)),
                regularization_db=float(data.get("regularization_db", -40.0)),
                smooth_octave=float(data.get("smooth_octave", 12.0)),
                max_side_gain_db=float(data.get("max_side_gain_db", 6.0)),
                ear_order=data.get("ear_order", "auto"),
                itd_mode=data.get("itd_mode", "auto"),
                hf_blend_start_hz=float(data.get("hf_blend_start_hz", 3500.0)),
                hf_blend_end_hz=float(data.get("hf_blend_end_hz", 9000.0)),
                tail_fraction=float(data.get("tail_fraction", 0.15)),
                strengths=strengths,
                remap=data.get("remap", "normal"),
                no_csv=False,
                verbose=False,
            )

        warn_messages = [str(w.message) for w in warns]

        # Формируем список результатов
        outputs = []
        for entry in meta["outputs"]:
            wav_file = Path(entry["wav_file"])
            csv_file = Path(entry["response_csv"]) if entry["response_csv"] else None

            outputs.append({
                "strength": entry["strength"],
                "token": strength_token(entry["strength"]),
                "wav_filename": wav_file.name,
                "csv_filename": csv_file.name if csv_file else None,
                "latency_ms": entry["latency_ms"],
                "omitted_energy_db": entry["omitted_circular_energy_db"],
                "max_gain_db": entry["max_matrix_frequency_gain_db"],
                "recommended_preamp_db": entry["recommended_preamp_db"],
                "actual_peak_gain_db": entry["actual_peak_gain_db"],
                "mid_error": entry["mid_error_max_linear"],
                "side_gain_min_db": entry["actual_side_gain_min_db"],
                "side_gain_max_db": entry["actual_side_gain_max_db"],
            })

        meta_filename = Path(f"{out_prefix}_df_midsafe_meta.json").name

        result = {
            "session_id": session_id,
            "outputs": outputs,
            "meta_filename": meta_filename,
            "channel_order": meta["channel_order"],
            "fs": meta["fs"],
            "warnings": warn_messages,
            "matrix_definition": meta["matrix_definition"],
        }

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Pipeline error")
        return jsonify({"error": f"Pipeline error: {e}"}), 500


@app.route("/api/csv/<session_id>/<filename>")
def get_csv(session_id, filename):
    """Возвращает CSV как JSON для построения графиков."""
    session_dir = WORK_DIR / session_id
    if not session_dir.is_dir():
        return jsonify({"error": "Session not found"}), 404

    csv_path = session_dir / "out" / filename
    if not csv_path.is_file():
        return jsonify({"error": "File not found"}), 404

    import numpy as np
    data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)

    result = {
        "frequency": data[:, 0].tolist(),
        "ipsi_db": data[:, 1].tolist(),
        "contra_db": data[:, 2].tolist(),
        "target_side_db": data[:, 3].tolist(),
        "target_side_phase": data[:, 4].tolist(),
        "actual_side_db": data[:, 5].tolist(),
        "actual_side_phase": data[:, 6].tolist(),
        "actual_mid_db": data[:, 7].tolist(),
        "actual_side_raw_db": data[:, 8].tolist(),
    }

    return jsonify(result)


@app.route("/api/download/<session_id>/<filename>")
def download_file(session_id, filename):
    """Скачивание WAV/CSV/JSON файла."""
    session_dir = WORK_DIR / session_id
    if not session_dir.is_dir():
        return jsonify({"error": "Session not found"}), 404

    file_path = session_dir / "out" / filename
    if not file_path.is_file():
        return jsonify({"error": "File not found"}), 404

    return send_file(str(file_path), as_attachment=True, download_name=filename)


@app.route("/api/meta/<session_id>/<filename>")
def get_meta(session_id, filename):
    """Возвращает метаданные JSON."""
    session_dir = WORK_DIR / session_id
    if not session_dir.is_dir():
        return jsonify({"error": "Session not found"}), 404

    meta_path = session_dir / "out" / filename
    if not meta_path.is_file():
        return jsonify({"error": "File not found"}), 404

    with open(meta_path, encoding="utf-8") as f:
        return jsonify(json.load(f))


def run_web(host="127.0.0.1", port=5173, debug=False):
    """Запуск веб-сервера."""
    print(f"HRTF Crossfeed Web UI: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_web()
