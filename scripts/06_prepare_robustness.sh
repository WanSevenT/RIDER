#!/usr/bin/env bash
set -euo pipefail

# Prepare the 11 perturbation settings used by the Diffusion/T2I-8 robustness
# protocol. The historical Gaussian-noise realization used for the paper did
# not record an RNG seed. SEED=42 provides a deterministic protocol replica.
SRC_ROOT=${SRC_ROOT:-./dataset/test}
DST_ROOT=${DST_ROOT:-./dataset/robust}
SEED=${SEED:-42}

run_attack() {
  local name=$1
  local corruption=$2
  local strength=$3
  python tools/make_robust_testsets.py \
    --src_root "$SRC_ROOT" \
    --dst_root "$DST_ROOT/$name" \
    --corruption "$corruption" \
    --strength "$strength" \
    --seed "$SEED" \
    --include_top_dirs ldm sd1.4 sd2.1 sdxl sd3.5 flux gpt-image1.5 gpt-image2.0 \
    --skip_existing
}

run_attack jpeg95    jpeg   95
run_attack jpeg75    jpeg   75
run_attack jpeg50    jpeg   50
run_attack resize075 resize 0.75
run_attack resize05  resize 0.50
run_attack blur3     blur   3
run_attack blur5     blur   5
run_attack noise5    noise  5
run_attack noise10   noise  10
run_attack crop09    crop   0.90
run_attack crop075   crop   0.75
