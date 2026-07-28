#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python"
QUICK=""; MODEL="assets/sample.glb"; OUT="output"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK="--quick"; shift;;
    --model) MODEL="$2"; shift 2;;
    --out)   OUT="$2"; shift 2;;
    *) echo "unknown: $1"; exit 1;;
  esac
done
echo "=== 1. pipeline ==="
eval $PY script/run_pipeline.py --model "$MODEL" --out "$OUT" $QUICK --solid-union-res 512 --octree-depth 9 --iters 10000 --batch 65536
echo "=== 2. rgbcrit ==="
eval $PY script/rerender_rgbcrit.py --out "$OUT"
echo "=== 3. res512 ==="
eval $PY script/rerender_res512.py --out "$OUT"
echo "=== 4. showcase ==="
eval $PY script/rerender_showcase.py --out "$OUT"
echo "=== DONE: $OUT ==="
