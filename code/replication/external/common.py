"""Month arithmetic and the error type of the external-control analysis."""
from __future__ import annotations

import numpy as np


class ExternalControlError(RuntimeError):
    """An input or consistency check of the external-control analysis failed (nothing is written after it)."""


def month_index(label) -> int:
    """'YYYY-MM' -> year * 12 + month - 1."""
    text = str(label)
    return int(text[:4]) * 12 + int(text[5:7]) - 1


def month_label(index: int) -> str:
    return f'{int(index) // 12:04d}-{int(index) % 12 + 1:02d}'


def months(start: str, end: str) -> list:
    return [month_label(i) for i in range(month_index(start), month_index(end) + 1)]


def ym_index(year, month) -> np.ndarray:
    return np.asarray(year, np.int64) * 12 + np.asarray(month, np.int64) - 1
