#!/usr/bin/env bash
set -euo pipefail
python experiments/artifact_ablation/train_artifact_ablation.py \
  --train_root ./dataset/train \
  --val_root ./dataset/val \
  --test_root ./dataset/test \
  --checkpoint ./outputs/ablations/artifact/spectral_haar_ms_npr/best.pth \
  --epochs 20 \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 8 \
  --device cuda \
  --seed 42 \
  --base_lr 1e-4 \
  --min_lr 1e-6 \
  --scheduler cosine \
  --checkpoint_metric macro_accuracy \
  --threshold_metric macro_accuracy \
  --hard_mining_dynamic_boost 2.0 \
  --artifact_supcon_weight 0.05 \
  --artifact_domain_adv_weight 0.05 \
  --artifact_branches spectral_mag wavelet npr \
  --npr_scales 0.25 0.5 0.75 \
  --use_amp \
  --amp_dtype bf16 \
  --save_test_reports
