#!/bin/bash
# Run on RTX 5090: train → shutdown (no git)
# Usage: nohup bash scripts/train_eval_5090.sh > train.log 2>&1 &
set -eo pipefail

# Prevent CUDA fragmentation OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# On any error: log, shutdown (no git)
on_error() {
    echo "=== ERROR at $(date) ===" | tee -a error.log
    echo "Exit code: $?" >> error.log
    tail -50 train.log >> error.log 2>/dev/null || true
    # git add error.log && git commit -m "Auto shutdown after error — $(date -Isec)" && git push || true
    # AUTO-SHUTDOWN DISABLED FOR DEBUGGING
    # /usr/bin/autodl shutdown 2>/dev/null || shutdown -h now 2>/dev/null || poweroff
    exit 1
}
trap on_error ERR

MODEL_PATH="${1:-./Qwen3-0.6B}"
OUT_DIR="out/sid_generator"

echo "=== Phase 1: Training ==="
python src/sid/train_sid_generator.py \
  --train_pt data/sid_train_512.pt \
  --eval_pt data/sid_eval_512.pt \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_DIR" \
  --preset 5090 \
  --epochs 3

echo "=== Done ==="
# AUTO-SHUTDOWN DISABLED FOR DEBUGGING
# /usr/bin/autodl shutdown 2>/dev/null || shutdown -h now 2>/dev/null || poweroff
