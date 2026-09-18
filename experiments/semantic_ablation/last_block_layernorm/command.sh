#!/usr/bin/env bash
set -euo pipefail
python src/train_semantic.py \
  --train_root ./dataset/train \
  --val_root ./dataset/val \
  --test_root ./dataset/test \
  --checkpoint ./outputs/ablations/semantic/last_block_layernorm/best.pth \
  --epochs 20 \
  --batch_size 32 \
  --num_workers 8 \
  --device cuda \
  --seed 42 \
  --clip_lr 5e-7 \
  --base_lr 1e-4 \
  --clip_finetune_last_n 1 \
  --patch_shuffle_prob 0.0 \
  --semantic_dropout 0.0 \
  --semantic_token_mask_prob 0.0 \
  --hard_mining_dynamic_boost 0.0 \
  --threshold_metric macro_balanced_acc \
  --use_amp \
  --amp_dtype bf16 \
  --save_test_reports \
  --save_test_interval 5
