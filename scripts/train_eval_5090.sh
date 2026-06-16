#!/bin/bash
# Run on RTX 5090: train → infer → eval → git push → shutdown
# Usage: nohup bash scripts/train_eval_5090.sh > train.log 2>&1 &
#
# Before first run, set up git token auth once:
#   git config credential.helper store
#   git push https://Goodnamee:TOKEN@github.com/Goodnamee/music-rec.git master
#   (TOKEN: GitHub Settings → Developer settings → Personal access tokens → Tokens (classic) → repo scope)
set -euo pipefail

OUT_DIR="out/sid_generator"
EXP_DIR="exp/inference/devset"

echo "=== Phase 1: Training ==="
python src/sid/train_sid_generator.py \
  --train_pt data/sid_train_512.pt \
  --eval_pt data/sid_eval_512.pt \
  --model_id Qwen/Qwen3-0.6B \
  --output_dir "$OUT_DIR" \
  --preset 5090 \
  --epochs 5

echo "=== Phase 2: Inference ==="
python src/sid/sid_inference.py \
  --model_dir "$OUT_DIR" \
  --sid_to_tracks exp/sid/rqvae_2176d_d4_k256/sid_to_tracks.json \
  --track_to_sid exp/sid/rqvae_2176d_d4_k256/track_to_sid.json \
  --out "$EXP_DIR/sid_generator.json"

echo "=== Phase 3: Evaluation ==="
python src/evaluate.py \
  --inference "$EXP_DIR/sid_generator.json" \
  --scores exp/scores/devset/sid_generator.json \
  --ground_truth exp/ground_truth/devset.json

echo "=== Results ==="
cat exp/scores/devset/sid_generator.json

echo "=== Phase 4: Git push ==="
git add exp/inference/devset/sid_generator.json exp/scores/devset/sid_generator.json
git commit -m "SID Generator 5090 results — $(date -I)" || true
git push

echo "=== Done, shutting down ==="
sudo shutdown -h now
