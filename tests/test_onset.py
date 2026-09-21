"""Тесты onset-детекции и выравнивания HRIR."""

import numpy as np

from hrtf_crossfeed.onset import extract_onset_aligned, onset_index


def test_onset_empty():
    assert onset_index(np.array([])) == 0


def test_onset_silent():
    """Тихий сигнал → onset = 0."""
    h = np.zeros(100)
    assert onset_index(h) == 0


def test_onset_impulse_at_start():
    """Импульс в начале → onset = 0 или около."""
    h = np.zeros(64)
    h[0] = 1.0
    idx = onset_index(h, threshold_db=-24.0)
    assert idx <= 4


def test_onset_impulse_delayed():
    """Импульс в середине → onset около позиции импульса."""
    h = np.zeros(128)
    onset_pos = 40
    h[onset_pos] = 1.0
    # Немного предзвона для устойчивости
    h[onset_pos - 1] = 0.1

    idx = onset_index(h, threshold_db=-24.0)
    assert abs(idx - onset_pos) <= 5


def test_onset_threshold_higher_later():
    """Более высокий порог → onset позже (или тот же)."""
    h = np.zeros(128)
    h[30] = 0.3
    h[50] = 1.0

    idx_low = onset_index(h, threshold_db=-40.0)
    idx_high = onset_index(h, threshold_db=-6.0)

    # При низком пороге onset раньше, при высоком — позже (или равно)
    assert idx_low <= idx_high


def test_extract_onset_aligned_basic():
    """Onset выравнивается точно на индекс pre."""
    h = np.zeros(64)
    onset = 20
    h[onset] = 1.0
    h[onset + 1] = 0.5

    pre = 5
    out_len = 32

    y = extract_onset_aligned(h, onset, pre, out_len)

    assert len(y) == out_len
    # Основной пик должен быть на индексе pre
    assert np.argmax(np.abs(y)) == pre


def test_extract_onset_aligned_pre_zero():
    """pre=0 → onset на индексе 0."""
    h = np.zeros(32)
    h[10] = 1.0

    y = extract_onset_aligned(h, 10, pre=0, out_len=32)

    assert len(y) == 32
    assert np.argmax(np.abs(y)) == 0


def test_extract_onset_aligned_tail_fade():
    """Хвост выходного FIR плавно затухает к нулю."""
    h = np.ones(128)
    onset = 10

    y = extract_onset_aligned(h, onset, pre=5, out_len=64, tail_fraction=0.25)

    assert len(y) == 64
    # Последний сэмпл должен быть близок к нулю
    assert abs(y[-1]) < 0.1


def test_extract_onset_aligned_clipping():
    """Если исходный HRIR длиннее выходного — обрезка без ошибки."""
    h = np.ones(256)
    onset = 50

    y = extract_onset_aligned(h, onset, pre=10, out_len=32)

    assert len(y) == 32
