#!/usr/bin/env bash
# Launcher: checks dependencies (keeps the platform's own PyTorch build) and
# the base model, then runs main.py with the given arguments.  The Dolly data
# ships with the code (data/databricks-dolly-15k.jsonl).
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # the model is read from a local folder
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PIP_PKGS="$PROJECT_DIR/pip_pkgs"
export PYTHONPATH="$PIP_PKGS${PYTHONPATH:+:$PYTHONPATH}"
check='import torch, numpy, pandas, sklearn, scipy, openpyxl, matplotlib, transformers, safetensors
v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
assert (4, 45) <= v < (5, 0), transformers.__version__'
if ! "$PYTHON_BIN" -c "$check" 2>/dev/null; then
    echo "===== 安装依赖到 pip_pkgs/（保留平台自带 PyTorch） ====="
    "$PYTHON_BIN" -c "import torch" || { echo "需要先有 PyTorch" >&2; exit 1; }
    mkdir -p "$PIP_PKGS"
    "$PYTHON_BIN" -m pip install --requirement requirements-cloud.txt \
        --target "$PIP_PKGS" --no-cache-dir --upgrade ${PIP_INDEX_URL:+-i "$PIP_INDEX_URL"}
    "$PYTHON_BIN" -c "$check"
fi
"$PYTHON_BIN" - <<'PY'
import torch, numpy, pandas, sklearn, transformers
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"transformers={transformers.__version__} numpy={numpy.__version__} "
      f"pandas={pandas.__version__} sklearn={sklearn.__version__}")
PY

MODEL_PATH="${MODEL_PATH:-model_cache/Qwen2.5-0.5B}"
if [ ! -f "$MODEL_PATH/config.json" ] || [ ! -f "$MODEL_PATH/model.safetensors" ]; then
    echo "找不到基座模型 $MODEL_PATH。先运行: bash start.sh fetch-model" >&2
    echo "(or place config.json, model.safetensors and the tokenizer files there by hand)" >&2
    exit 1
fi

exec "$PYTHON_BIN" main.py --model-path "$MODEL_PATH" "$@"
