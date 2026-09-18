#!/usr/bin/env bash
set -euo pipefail
python src/train_artifact.py \
  --train_root ./dataset/train \
  --val_root ./dataset/val \
  --test_root ./dataset/test \
  --checkpoint ./checkpoints/artifact/artifact_best_full.pth \
  --epochs 20 \
  --image_size 224 \
  --batch_size 16 \
  --num_workers 8 \
  --base_lr 1e-4 \
  --checkpoint_metric macro_accuracy \
  --threshold_metric macro_accuracy \
  --artifact_branches spectral_mag wavelet npr \
  --npr_scales 0.25 0.5 0.75 \
  --normalized_artifact_inputs \
  --artifact_supcon_weight 0.05 \
  --artifact_domain_adv_weight 0.05 \
  --hard_mining_dynamic_boost 2.0 \
  --save_test_reports \
  --periodic_eval_interval 1 \
  --show_progress
