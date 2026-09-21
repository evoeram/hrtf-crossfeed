"""Общие утилиты: преобразования dB, нормализация, плавные затухания."""

import numpy as np

EPS = 1e-15


def db(x, floor_db=-240.0):
    """Преобразование линейной величины в децибелы с нижним порогом."""
    x = np.asarray(x)
    magnitude = np.asarray(np.abs(x), dtype=np.float64)
    floor = 10.0 ** (floor_db / 20.0)

    return 20.0 * np.log10(
        np.maximum(magnitude, floor)
    )


def undb(x_db):
    """Обратное преобразование децибел в линейную величину."""
    return 10.0 ** (np.asarray(x_db, dtype=np.float64) / 20.0)


def wrap_deg(a):
    """Заворачивает углы в диапазон [-180, 180)."""
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def ensure_len(x, n):
    """Дополняет или обрезает массив до длины n."""
    x = np.asarray(x, dtype=np.float64)

    if len(x) < n:
        return np.pad(x, (0, n - len(x)))

    return x[:n]


def rms(x):
    """Среднеквадратичное значение."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + EPS))


def safe_attr(obj, name, default=""):
    """Безопасное чтение атрибута netCDF variable, с декодированием bytes."""
    try:
        value = getattr(obj, name)
    except Exception:
        return default

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def validate_finite(name, x):
    """Выбрасывает ValueError, если массив содержит NaN или Inf."""
    x = np.asarray(x)

    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or Inf")


def tail_fade(x, fraction=0.15):
    """
    Оставляет начало FIR неизменным и плавно сводит к нулю хвост.

    fraction — доля длины, занимаемая косинусным затуханием.
    """
    x = np.asarray(x, dtype=np.float64).copy()

    if fraction <= 0.0 or len(x) < 4:
        return x

    n = max(2, int(round(len(x) * fraction)))
    n = min(n, len(x))

    fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n)))
    x[-n:] *= fade

    return x
