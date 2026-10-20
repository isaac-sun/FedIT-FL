#!/usr/bin/env bash
# Download the base model (default Qwen/Qwen2.5-0.5B, Apache-2.0, ~1 GB) into
# model_cache/<name>.  Mirrors are tried in order: ModelScope (fast inside
# mainland China, e.g. AutoDL), hf-mirror.com, huggingface.co.  Finished
# files are kept, so rerunning resumes.
#   bash fetch_model.sh                      # Qwen/Qwen2.5-0.5B
#   MODEL_ID=Qwen/Qwen2.5-1.5B bash fetch_model.sh
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B}"
DEST="${MODEL_PATH:-model_cache/${MODEL_ID##*/}}"
FILES="config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt LICENSE model.safetensors"
mkdir -p "$DEST"

urls() {   # candidate URLs of one file, best first
    echo "https://www.modelscope.cn/models/$MODEL_ID/resolve/master/$1"
    echo "https://www.modelscope.cn/api/v1/models/$MODEL_ID/repo?Revision=master&FilePath=$1"
    echo "https://hf-mirror.com/$MODEL_ID/resolve/main/$1"
    echo "https://huggingface.co/$MODEL_ID/resolve/main/$1"
}

for file in $FILES; do
    if [ -s "$DEST/$file" ]; then echo "have $DEST/$file"; continue; fi
    ok=0
    while read -r url; do
        echo "get  $file <- ${url%%\?*}"
        if curl -fL --retry 3 --connect-timeout 20 -C - -o "$DEST/$file.part" "$url"; then
            # A mirror may answer with an HTML error page instead of the file.
            if head -c 200 "$DEST/$file.part" | grep -qi "<html"; then
                rm -f "$DEST/$file.part"; continue
            fi
            mv "$DEST/$file.part" "$DEST/$file"; ok=1; break
        fi
        # keep a partial download: every mirror serves the same bytes, so the
        # next mirror (or a rerun) continues it with "-C -"
    done < <(urls "$file")
    if [ "$ok" = 0 ]; then
        [ "$file" = LICENSE ] && continue   # optional
        echo "could not download $file; download it by hand into $DEST/" >&2
        exit 1
    fi
done

PYTHONPATH="$PROJECT_DIR/pip_pkgs${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON_BIN:-python3}" - "$DEST" <<'PY'
import json, os, sys
dest = sys.argv[1]
config = json.load(open(os.path.join(dest, "config.json")))
size = os.path.getsize(os.path.join(dest, "model.safetensors"))
with open(os.path.join(dest, "model.safetensors"), "rb") as handle:
    header = int.from_bytes(handle.read(8), "little")
    tensors = json.loads(handle.read(header))
print(f"ok: {dest} ({config.get('model_type')}, {config.get('num_hidden_layers')} layers, "
      f"{len(tensors) - ('__metadata__' in tensors)} tensors, {size / 1e9:.2f} GB)")
PY
