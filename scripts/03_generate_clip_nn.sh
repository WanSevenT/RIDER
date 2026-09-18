#!/usr/bin/env bash
set -euo pipefail
python src/generate_clip_nn_scores.py \
  --train_root ./dataset/train \
  --val_root ./dataset/val \
  --test_root ./dataset/test \
  --output_dir ./checkpoints/clip_nn_vitl14_k10_m500 \
  --clip_model_name ViT-L/14 \
  --batch_size 128 \
  --num_workers 8 \
  --k 10 \
  --max_per_model_class 500 \
  --tau 40.0 \
  --bias 0.0
