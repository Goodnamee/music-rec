#!/bin/bash
# Run on RTX 5090: train → infer → eval → git push → shutdown
# Usage: nohup bash scripts/train_eval_5090.sh > train.log 2>&1 &
#
# Set up git auth once before running:
#   git config credential.helper store
#   git push https://Goodnamee:TOKEN@github.com/Goodnamee/music-rec.git master
set -eo pipefail

# Prevent CUDA fragmentation OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# On any error: log, push, shutdown
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
EXP_DIR="exp/inference/devset"
SCORE_DIR="exp/scores/devset"

echo "=== Phase 1: Training ==="
python src/sid/train_sid_generator.py \
  --train_pt data/sid_train_3tok_512.pt \
  --eval_pt data/sid_eval_3tok_512.pt \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_DIR" \
  --preset 5090 \
  --epochs 3

echo "=== Phase 2: Inference ==="
python src/sid/sid_inference.py \
  --model_dir "$OUT_DIR" \
  --model_path "$MODEL_PATH" \
  --sid_to_tracks exp/sid/rqvae_2176d_d4_k256/sid_to_tracks_3tok.json \
  --track_to_sid exp/sid/rqvae_2176d_d4_k256/track_to_sid_3tok.json \
  --out "$EXP_DIR/sid_generator.json"

echo "=== Phase 3: Evaluation ==="
mkdir -p "$SCORE_DIR"
python src/evaluate.py \
  --inference "$EXP_DIR/sid_generator.json" \
  --scores "$SCORE_DIR/sid_generator.json" \
  --ground_truth exp/ground_truth/devset.json

echo "=== Results ==="
cat "$SCORE_DIR/sid_generator.json"

echo "=== Git push results ==="
git add "$EXP_DIR/sid_generator.json" "$SCORE_DIR/sid_generator.json"
git commit -m "SID Generator results — $(date -Isec)" || true
git push

echo "=== Done ==="
# AUTO-SHUTDOWN DISABLED FOR DEBUGGING
# /usr/bin/autodl shutdown 2>/dev/null || shutdown -h now 2>/dev/null || poweroff
