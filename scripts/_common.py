"""Shared set-up of the numbered scripts: the package code on the path, the pinned BLAS thread count before numpy
loads, and offline model loading."""
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'code'))
# Bit-identical reruns: the BLAS thread count is pinned before numpy loads (see README, "随机种子与可重复性"). The
# reference results were computed with 4 threads for the main analysis and with 12 threads for the external-control
# analysis (script 07); these defaults reproduce them bit for bit. Other thread counts change results only in the last
# digits (see README section 7); REPLICATION_BLAS_THREADS overrides the default of every script. The thread variables
# are always set here (inherited values are overridden).
DEFAULT_THREADS = {'07_external_control_table9.py': '12'}
_script = Path(sys.argv[0]).name
_threads = os.environ.get('REPLICATION_BLAS_THREADS') or DEFAULT_THREADS.get(_script, '4')
for _var in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_var] = _threads
print(f'[{Path(_script).stem}] BLAS threads: {_threads}', flush=True)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
