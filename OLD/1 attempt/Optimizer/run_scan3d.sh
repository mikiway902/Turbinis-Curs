#!/usr/bin/env bash
# Этап 3: полная сетка (n, D_вт, w2) 31^3 для конфигураций A и B.
# Параллельно через ProcessPoolExecutor (--workers), затем слияние и отчёт.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

W=8
STEPS=31
rm -f results/scan3d_A.csv results/scan3d_B.csv results/SCAN3D_DONE

python -u scan3d.py --config A --workers "$W" --steps "$STEPS" > results/scan3d_A.log 2>&1
python -u scan3d.py --config B --workers "$W" --steps "$STEPS" > results/scan3d_B.log 2>&1
python -u scan3d.py --merge --steps "$STEPS" > results/scan3d_merge.log 2>&1
echo DONE > results/SCAN3D_DONE
