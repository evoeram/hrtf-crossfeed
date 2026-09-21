"""
hrtf-crossfeed
==============

Построение complex HRTF-derived, mid-safe virtual-speaker crossfeed матриц
из SOFA HRTF/DTF файлов.

Установка
---------
    pip install -e .

Использование
-------------
    python -m hrtf_crossfeed --sofa input.sofa --out-prefix out/ari1059

Или с расширенными параметрами::

    python -m hrtf_crossfeed \\
        --sofa input.sofa \\
        --out-prefix out/ari1059 \\
        --az 30 \\
        --ir-len 512 \\
        --latency-samples 128 \\
        --strengths 0.70,1.00,1.20

Программный API
---------------
.. code-block:: python

    from hrtf_crossfeed import run_pipeline

    meta = run_pipeline(
        sofa_path="input.sofa",
        out_prefix="out/ari1059",
        strengths=[1.0],
    )

Модули
------
- ``utils``        — dB/undb, tail_fade, валидация
- ``sofa``         — загрузка SOFA (netCDF4)
- ``geometry``     — координаты SourcePosition, поиск ближайшего
- ``delay``        — обработка Data.Delay
- ``ear_order``    — определение порядка ушей
- ``onset``        — onset-детекция и выравнивание HRIR
- ``smoothing``    — octave magnitude сглаживание, limit gain
- ``itd``          — Woodworth ITD модель и валидация
- ``side_response`` — построение Side transfer
- ``matrix``       — построение mid-safe FIR матрицы
- ``io``           — remap, сохранение WAV/CSV
- ``pipeline``     — высокоуровневый пайплайн
- ``cli``          — command-line интерфейс
"""

__version__ = "1.0.0"

from .matrix import build_midsafe_matrix_from_side
from .pipeline import parse_strengths, run_pipeline, strength_token
from .side_response import build_target_side_response

__all__ = [
    "__version__",
    "run_pipeline",
    "parse_strengths",
    "strength_token",
    "build_midsafe_matrix_from_side",
    "build_target_side_response",
    "utils",
    "sofa",
    "geometry",
    "delay",
    "ear_order",
    "onset",
    "smoothing",
    "itd",
    "side_response",
    "matrix",
    "io",
    "pipeline",
    "cli",
    "web",
]
