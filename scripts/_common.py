"""Shared set-up of the numbered scripts: the package code on the path, the BLAS thread count before numpy loads,
and offline model loading.
"""
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'code'))
# BLAS threads are pinned before numpy loads: 4 for the main analysis, 12 for script 07
# (REPLICATION_BLAS_THREADS overrides)
DEFAULT_THREADS = {'07_external_control_table9.py': '12'}
_script = Path(sys.argv[0]).name
_threads = os.environ.get('REPLICATION_BLAS_THREADS') or DEFAULT_THREADS.get(_script, '4')
for _var in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_var] = _threads
print(f'[{Path(_script).stem}] BLAS threads: {_threads}', flush=True)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
