#!/usr/bin/env bash
set -euo pipefail
python src/train_semantic.py \
  --train_root ./dataset/train \
  --val_root ./dataset/val \
  --test_root ./dataset/test \
  --checkpoint ./checkpoints/semantic/semantic_best_full.pth \
  --epochs 5 \
  --batch_size 32 \
  --num_workers 8 \
  --clip_finetune_last_n 1 \
  --clip_lr 1e-5 \
  --base_lr 1e-7 \
  --hard_mining_dynamic_boost 0.0 \
  --semantic_dropout 0.0 \
  --semantic_token_mask_prob 0.0 \
  --patch_shuffle_prob 0.0 \
  --threshold_metric macro_balanced_acc \
  --use_amp \
  --amp_dtype bf16 \
  --save_test_reports \
  --save_test_interval 5 \
  --show_progress
