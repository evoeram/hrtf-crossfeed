# hrtf-crossfeed

Построение **complex HRTF-derived, mid-safe virtual-speaker crossfeed** матриц из SOFA HRTF/DTF файлов.

## Что делает

Генерирует 4-канальные FIR фильтры (LL, LR, RL, RR) для кроссфид-матрицы на основе измерений HRTF из SOFA-файлов. Матрица построена так, что Mid-канал всегда точно равен delayed identity, а Side-канал управляется через параметр `strength`:

- `strength = 0` — identity (без кроссфида)
- `strength = 1` — измеренный speaker target
- `strength > 1` — экстраполяция

## Установка

```bash
pip install -e .
```

Зависимости: `numpy`, `scipy`, `soundfile`, `netCDF4`

## Использование

### CLI

```bash
python -m hrtf_crossfeed \
    --sofa input.sofa \
    --out-prefix out/ari1059 \
    --az 30 \
    --ir-len 512 \
    --latency-samples 128 \
    --strengths 0.70,1.00,1.20
```

### Программно

```python
from hrtf_crossfeed.pipeline import run_pipeline

meta = run_pipeline(
    sofa_path="input.sofa",
    out_prefix="out/ari1059",
    strengths=[1.0],
    az=30.0,
    ir_len=512,
)
```

## Структура модулей

| Модуль | Назначение |
|---|---|
| `utils` | dB/undb, tail_fade, валидация, RMS |
| `sofa` | Загрузка SOFA (netCDF4) |
| `geometry` | Координаты SourcePosition, поиск ближайшего направления |
| `delay` | Обработка Data.Delay |
| `ear_order` | Определение порядка ушей (auto/normal/swapped) |
| `onset` | Onset-детекция и выравнивание HRIR |
| `smoothing` | Octave magnitude сглаживание, ограничение усиления |
| `itd` | Woodworth ITD модель и валидация |
| `side_response` | Построение целевого Side transfer |
| `matrix` | Построение mid-safe FIR матрицы |
| `io` | Remap каналов, сохранение WAV/CSV |
| `pipeline` | Высокоуровневый пайплайн (связывает всё вместе) |
| `cli` | Command-line интерфейс |

## Выходные файлы

Для каждого значения `strength`:

- `*_df_midsafe_s{token}.wav` — 4-канальный float32 WAV (LL, LR, RL, RR)
- `*_df_midsafe_s{token}_response.csv` — частотные отклики
- `*_df_midsafe_meta.json` — полные метаданные

## Ключевые параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--az` | 30 | Азимут виртуальных колонок (градусы) |
| `--ir-len` | 512 | Длина выходных FIR |
| `--latency-samples` | 128 | Общая задержка |
| `--strengths` | 0.70,1.00,1.20 | Сила кроссфида |
| `--ear-order` | auto | Порядок ушей в Data.IR |
| `--itd-mode` | auto | ITD: measured / woodworth / auto |
| `--smooth-octave` | 12 | Сглаживание (1/N octave) |
| `--max-side-gain-db` | 6 | Макс усиление Side response |
| `--hf-blend-start` | 3500 | Начало HF blend to identity |
| `--hf-blend-end` | 9000 | Конец HF blend (R→1) |

## Лицензия

MIT
