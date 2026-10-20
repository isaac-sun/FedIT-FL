#!/usr/bin/env bash
# Pilot: seed 42, IID and Dirichlet alpha=0.1, no attack + DFR/SDFR/AFR at
# 30% = 8 runs, every run scoring all Shapley axes.  Use the per-round times
# in round_curves.csv (and summary.json) to size the formal matrix.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
LOCAL_LR="${LOCAL_LR:-1e-4}"
case "$LOCAL_LR" in 1e-4|1e-04|0.0001) LR_TAG="" ;; *) LR_TAG="_lr$LOCAL_LR" ;; esac
if [ -n "${GLOBAL_SIZE:-}" ] && [ "$GLOBAL_SIZE" != 480 ]; then LR_TAG="${LR_TAG}_g$GLOBAL_SIZE"; fi
RESULTS_ROOT="${RESULTS_ROOT:-results/pilot$LR_TAG}"
DEVICE="${FORCE_DEVICE:-auto}"
COMMON=(--device "$DEVICE" --num-rounds 25 --num-clients 20 --participation-ratio 1.0
        --attack-types dfr,sdfr,afr --malicious-ratios 0.3
        --experiment-seeds 42 --num-mc-samples 20 --local-lr "$LOCAL_LR"
        ${SHAPLEY_AXES:+--shapley-axes "$SHAPLEY_AXES"}
        ${GLOBAL_SIZE:+--global-size "$GLOBAL_SIZE"}
        ${EVAL_TOKEN_BUDGET:+--eval-token-budget "$EVAL_TOKEN_BUDGET"}
        --summary-root "$RESULTS_ROOT")
bash run.sh "$@" "${COMMON[@]}" --partition-method iid --results-dir "$RESULTS_ROOT/iid"
bash run.sh "$@" "${COMMON[@]}" --partition-method dirichlet --dirichlet-alpha 0.1 \
    --results-dir "$RESULTS_ROOT/dirichlet_alpha0p1"
PYTHONPATH="$PROJECT_DIR/pip_pkgs${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON_BIN:-python3}" summarize.py --results-root "$RESULTS_ROOT"
