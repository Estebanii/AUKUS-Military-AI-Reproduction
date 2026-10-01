"""Writing and reading the result files under ``results/`` (JSON, CSV) and a timing / memory log per script."""
from __future__ import annotations

import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

from . import data


def plain(obj):
    """A JSON-ready copy: numpy scalars and arrays become Python numbers and lists; tuples become lists."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [plain(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _nan_safe(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ('Infinity' if obj > 0 else '-Infinity')
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_safe(v) for v in obj]
    return obj


def write_json(rel: str, obj) -> Path:
    path = data.results_dir() / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_nan_safe(plain(obj)), indent=1, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')
    return path


def read_json(rel: str) -> dict:
    path = data.results_dir() / rel
    if not path.is_file():
        raise FileNotFoundError(f'{path}: run the script that writes it first (see README)')
    return json.loads(path.read_text(encoding='utf-8'))


def write_text(rel: str, text: str) -> Path:
    path = data.results_dir() / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


class Timer:
    """Log the wall time and the peak resident memory of a script to results/logs/runtime.tsv."""

    def __init__(self, name: str):
        self.name, self.start = name, time.time()

    def done(self) -> None:
        seconds = time.time() - self.start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_gb = peak / 1e9 if sys.platform == 'darwin' else peak * 1024 / 1e9
        log = data.results_dir() / 'logs' / 'runtime.tsv'
        log.parent.mkdir(parents=True, exist_ok=True)
        new = not log.exists()
        with open(log, 'a', encoding='utf-8') as handle:
            if new:
                handle.write('script\tseconds\tpeak_rss_gb\n')
            handle.write(f'{self.name}\t{seconds:.1f}\t{peak_gb:.2f}\n')
        print(f'[{self.name}] done in {seconds:.0f} s, peak memory {peak_gb:.1f} GB', flush=True)
