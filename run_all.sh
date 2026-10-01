#!/usr/bin/env bash
# Run every numbered script in order, then compare.py. See README.md.
#
#   ./run_all.sh                 # default: all tables and figures (about 10 minutes on the reference machine)
#   ./run_all.sh --full          # also recompute the external-control design simulations and run the external-control
#                                # chain for all five encoders (about 1 hour more)
#
# Environment: PYTHON (default .venv/bin/python), REPLICATION_DATA (default data/bundle), REPLICATION_RESULTS (default
# results), REPLICATION_BLAS_THREADS (default: 4, and 12 for script 07; see README).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PYTHON="${PYTHON:-$HERE/.venv/bin/python}"
FULL=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -x "$PYTHON" ]; then
  echo "Python not found at $PYTHON (create the environment first, see README section 3, or set PYTHON)" >&2
  exit 2
fi
DATA="${REPLICATION_DATA:-$HERE/data/bundle}"
if [ ! -f "$DATA/MANIFEST.sha256" ]; then
  echo "data bundle not found at $DATA (see data/README.md, or set REPLICATION_DATA)" >&2
  exit 2
fi
RESULTS="${REPLICATION_RESULTS:-$HERE/results}"
mkdir -p "$RESULTS/logs"

echo "=== record_run"
"$PYTHON" scripts/record_run.py 2>&1 | tee "$RESULTS/logs/record_run.log"

run() {
  local name="$1"; shift
  echo "=== $name $*"
  "$PYTHON" "scripts/$name.py" "$@" 2>&1 | tee "$RESULTS/logs/$name.log"
}

run 00_prepare
run 01_sample_table2
run 02_h1_manova_table3
run 03_h2_did_tables4_6_fig3
run 04_h3_neighbours_fig4
run 05_cross_encoder_table8
run 06_distance_table7
if [ "$FULL" = 1 ]; then
  run 07_external_control_table9 --recompute-design --all-encoders
else
  run 07_external_control_table9
fi
run 08_figures_1_2

echo "=== compare.py"
"$PYTHON" compare.py 2>&1 | tee "$RESULTS/logs/compare.log"
