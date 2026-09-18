# Experiments

This directory keeps only the controlled ablation configurations represented in the current manuscript. It is intentionally separate from `results/`: `experiments/` contains launch/configuration records, while `results/` contains only three paper-level summary CSV files.

## Semantic ablations

- `semantic_ablation/frozen_clip/`
- `semantic_ablation/last_block_layernorm/`
- `semantic_ablation/last_two_blocks/`

These correspond to the Semantic Training Ablation block in `results/ablation_results.csv`.

## Artifact-cue ablations

- `artifact_ablation/spectral_haar/`
- `artifact_ablation/spectral_haar_single_npr/`
- `artifact_ablation/spectral_haar_ms_npr/`
- `artifact_ablation/large_six_cue/`

`train_artifact_ablation.py` is the all-branch experimental trainer used for these controlled cue-design runs. The public final RIDER artifact implementation remains `src/train_artifact.py`, which intentionally exposes only the selected three branches (`spectral_mag`, `wavelet`, and `npr`).

Generated checkpoints and logs are written under `outputs/` and are ignored by Git. The paper-aligned aggregate values are provided in `results/ablation_results.csv`.
