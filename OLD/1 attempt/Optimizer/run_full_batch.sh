#!/usr/bin/env bash
# Полный прогон: 8 сценариев параллельно, каждый — full10 (q_n<=0.995), затем qn1 (q_n<=1.0).
# По завершении — анализ (REPORT_full10.md, REPORT_qn1.md, REPORT_full10_vs_qn1.md).
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SCEN="A87_strict A87_relaxed A58_strict A58_relaxed B87_strict B87_relaxed B58_strict B58_relaxed"
COMMON="--seeds 10 --maxiter 10 --popsize 4"

run_one() {
    sc="$1"
    python -u optimize_extended.py --scenario "$sc" $COMMON --tag full10 > "results/${sc}_full10.log" 2>&1
    python -u optimize_extended.py --scenario "$sc" $COMMON --q-n 1.0 --tag qn1 > "results/${sc}_qn1.log" 2>&1
}

rm -f results/BATCH_DONE
for sc in $SCEN; do run_one "$sc" & done
wait

python -u analyze_extended.py --tags full10 qn1 > results/analysis.log 2>&1
echo DONE > results/BATCH_DONE
