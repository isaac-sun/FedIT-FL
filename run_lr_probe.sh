#!/usr/bin/env bash
# Learning-rate probe: one no-attack run per rate (Dirichlet alpha=0.5,
# seed 42, all 20 clients, 25 rounds), all in parallel on the GPU, Shapley
# off (fast).  lr_probe_report.py then prints the test loss by round, the
# share of the total loss drop taken by round 0, and a suggested rate.
# Usage: LRS="5e-5 1e-4 2e-4 5e-4" FORCE_DEVICE=cuda bash run_lr_probe.sh [--require-cuda]
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
LRS="${LRS:-5e-5 1e-4 2e-4 5e-4}"
ALPHA="${ALPHA:-0.5}"
ROOT="${RESULTS_ROOT:-results/lr_probe}"
DEVICE="${FORCE_DEVICE:-auto}"
THREADS=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu) / $(echo $LRS | wc -w) ))
[ "$THREADS" -ge 1 ] || THREADS=1
mkdir -p "$ROOT"
bash run.sh --help > /dev/null   # dependencies and model, once
pids=()
for LR in $LRS; do
    echo "===== lr=$LR -> $ROOT/lr_$LR (log $ROOT/lr_$LR.log) ====="
    bash run.sh "$@" --device "$DEVICE" --partition-method dirichlet --dirichlet-alpha "$ALPHA" \
        --num-rounds 25 --num-clients 20 --participation-ratio 1.0 \
        --attack-types "" --malicious-ratios "" --experiment-seeds 42 --no-shapley --no-plots \
        --local-lr "$LR" --torch-threads "$THREADS" --results-dir "$ROOT/lr_$LR" \
        > "$ROOT/lr_$LR.log" 2>&1 &
    pids+=($!)
done
fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
PYTHONPATH="$PROJECT_DIR/pip_pkgs${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON_BIN:-python3}" \
    lr_probe_report.py "$ROOT" || fail=1
exit "$fail"
