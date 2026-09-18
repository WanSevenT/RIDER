# RIDER: Reliability-Informed Dual-Expert Routing for AI-Generated Image Detection

RIDER is a two-stage detector for AI-generated images designed for cross-generator generalization and robustness to common image degradations. It combines a semantic forensic expert based on CLIP ViT-L/14 with an artifact forensic expert based on spectral magnitude, Haar wavelet, and multi-scale NPR cues. The experts are trained independently in Phase I. In Phase II, their logits are calibrated and combined by reliability-informed soft routing with CLIP-NN support.

The released checkpoint set corresponds to the reported result of **87.48% macro ACC / 95.26% macro AP** over the 20-dataset test benchmark.

## Results

| Evaluation | Macro ACC | Macro AP |
|---|---:|---:|
| Full test benchmark (20 datasets) | 87.48 | 95.26 |
| Unseen datasets (15 datasets) | 85.33 | 94.14 |
| Robustness average (11 perturbation settings) | 83.11 | 90.75 |
| Semantic expert | 85.02 | 93.82 |
| Artifact expert | 79.15 | 85.95 |

Detailed paper-level summaries are provided in:

```text
results/main_results.csv
results/ablation_results.csv
results/robustness_results.csv
```

## Repository structure

```text
RIDER/
├── environment_reference.txt  # Tested experiment environment
├── configs/        # Final configuration snapshot
├── datasets/       # Dataset documentation and exact sample manifests
├── experiments/    # Controlled ablation configurations
├── results/        # Paper-level result summaries
├── rider/          # Checkpoint loading utilities
├── scripts/        # Training, evaluation, and robustness entry points
├── src/            # Main implementation
├── tests/          # Repository validation tests
├── tools/          # Dataset, checkpoint, and robustness utilities
└── weights/        # Released compact checkpoints
```

## Installation

### Tested environment

The reported experiments were run with the following reference environment:

```text
Python:      3.12.0
PyTorch:     2.7.1+cu118
torchvision: 0.22.1+cu118
CUDA runtime: 11.8
```

The key package versions from the original experiment server are recorded in [`environment_reference.txt`](environment_reference.txt). `requirements.txt` pins the core Python dependencies used by RIDER, including Albumentations 2.0.6 and the exact OpenAI CLIP Git commit used in the experiments. PyTorch and torchvision are installed separately because their wheels are CUDA/platform specific.

A matching CUDA 11.8 environment can be created with:

```bash
conda create -n rider python=3.12 -y
conda activate rider
python -m pip install -U pip

pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

Development tests additionally require:

```bash
pip install -r requirements-dev.txt
```

Other CUDA or CPU builds may work, but the environment above is the tested configuration corresponding to the reported experiments. GPU/CUDA/cuDNN differences can introduce small run-to-run differences during retraining; the released checkpoints are the reference models for the reported numbers.

## Dataset preparation

The original benchmark images are **not redistributed** in this repository. RIDER uses a unified benchmark assembled from public sources. The exact retained sample set is specified by:

```text
datasets/manifests/dataset_manifest.csv
```

Each manifest entry records the split, benchmark source, model, label, relative path, file size, and SHA-256 hash.

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 79,987 | 80,000 | 159,987 |
| Validation | 1,597 | 1,600 | 3,197 |
| Test | 60,677 | 60,707 | 121,384 |
| **Total** | **142,261** | **142,307** | **284,568** |

Training and validation contain five generator groups: LDM, ProGAN, SD1.4, SD2.1, and SDXL. The test split contains 20 dataset/model groups: BigGAN, CRN, CycleGAN, DeepFake, FLUX, GauGAN, GPT-Image 1.5, GPT-Image 2.0, IMLE, LDM, ProGAN, SAN, SD1.4, SD2.1, SD3.5, SDXL, SeeingDark, StarGAN, StyleGAN, and StyleGAN2.

See [`datasets/README.md`](datasets/README.md) for source links and preparation notes. The exact train/validation/test directory snapshot used on the experiment server is also recorded in [`datasets/dataset_layout.txt`](datasets/dataset_layout.txt), with per-directory image counts in [`datasets/manifests/dataset_directory_counts.csv`](datasets/manifests/dataset_directory_counts.csv).

After arranging the data under `./dataset`, verify file presence:

```bash
python tools/verify_dataset.py --root ./dataset
```

For exact SHA-256 verification:

```bash
python tools/verify_dataset.py \
  --root ./dataset \
  --check_sha256
```

## Released checkpoints

The compact checkpoints used for the reported RIDER model are included in `weights/` and must be applied in the following order:

```text
OpenAI CLIP ViT-L/14
        ↓
semantic_best.pth
        ↓
artifact_best.pth
        ↓
fusion_best.pth
```

`fusion_best.pth` must be applied last because it contains the Phase-II-updated artifact BatchNorm running buffers in addition to fusion-specific parameters.

```text
weights/semantic_best.pth
weights/artifact_best.pth
weights/fusion_best.pth
```

Verify the checkpoint files on Linux/macOS with:

```bash
(cd weights && sha256sum -c checksums.sha256)
```

## Evaluation with released checkpoints

First generate the CLIP-NN scores:

```bash
bash scripts/03_generate_clip_nn.sh
```

Then evaluate the released RIDER checkpoint:

```bash
bash scripts/05_eval_rider.sh
```

Generated evaluation outputs are written to `outputs/eval/` and are ignored by Git. The reported ACC protocol uses a fixed decision threshold of 0.5.

## Training

### Phase I: semantic expert

```bash
bash scripts/01_train_semantic.sh
```

Final semantic configuration:

```text
Backbone:              CLIP ViT-L/14
Input resolution:      224 × 224
Trainable CLIP layers: final visual Transformer block + visual LayerNorm
Epochs:                5
Patch shuffle:         disabled
```

### Phase I: artifact expert

```bash
bash scripts/02_train_artifact.sh
```

Final artifact configuration:

```text
Input resolution:      224 × 224
Artifact cues:         spectral magnitude + Haar wavelet + MS-NPR
NPR scales:            0.25, 0.5, 0.75
Epochs:                20
Artifact inputs:       normalized
```

### CLIP-NN scores

```bash
bash scripts/03_generate_clip_nn.sh
```

Final CLIP-NN configuration:

```text
Backbone:                       CLIP ViT-L/14
k:                              10
Maximum samples/model/class:    500
tau:                            40
bias:                           0
```

### Phase II with released Phase-I experts

To reproduce the Phase-II training stage while keeping the released Phase-I experts fixed at their published checkpoints:

```bash
bash scripts/04_train_rider.sh
```

### Full retraining pipeline

To retrain the Phase-I experts first and then use those newly trained checkpoints in Phase II:

```bash
bash scripts/01_train_semantic.sh
bash scripts/02_train_artifact.sh
bash scripts/03_generate_clip_nn.sh
bash scripts/04_train_rider_from_scratch.sh
```

During Phase II, gradient updates to the loaded expert learnable parameters are disabled. Artifact BatchNorm running statistics can still update and are therefore stored in the final fusion checkpoint.

The complete public configuration snapshot is in:

```text
configs/final_config.yaml
```

## Ablation experiments

Controlled ablation configurations corresponding to the manuscript are provided under:

```text
experiments/semantic_ablation/
experiments/artifact_ablation/
```

The aggregate reported values are stored in:

```text
results/ablation_results.csv
```

Included semantic-training variants are Frozen CLIP, Last Block + LayerNorm, and Last Two Blocks. Included artifact-cue variants are Spectral + Haar, Spectral + Haar + Single-Scale NPR, Spectral + Haar + MS-NPR, and the Large Six-Cue Model.

Generated ablation checkpoints, logs, and per-image outputs are not committed.

## Robustness evaluation

The robustness benchmark uses Diffusion/T2I-8: LDM, SD1.4, SD2.1, SDXL, SD3.5, FLUX, GPT-Image 1.5, and GPT-Image 2.0. Eleven perturbation settings are evaluated across five perturbation families:

```text
JPEG quality:          95, 75, 50
Resize ratio:          0.75, 0.50
Gaussian blur setting: 3, 5
Gaussian noise sigma:  5, 10
Center-crop ratio:     0.90, 0.75
```

The released perturbation generator passes blur settings 3 and 5 to Pillow's `ImageFilter.GaussianBlur(radius=...)` implementation.

Utilities are provided in:

```text
tools/make_robust_testsets.py
tools/summarize_robust_metrics.py
```

A convenience script selects the eight robustness groups directly from `dataset/test` and prepares all 11 settings:

```bash
SRC_ROOT=./dataset/test \
DST_ROOT=./dataset/robust \
SEED=42 \
bash scripts/06_prepare_robustness.sh
```

The historical robustness run did not record a NumPy RNG seed for the Gaussian-noise image generation step. The `--seed` option in the public utility therefore provides a deterministic protocol replica, but regenerated noise images are not guaranteed to be bitwise identical to the historical noise realization used for the reported table. JPEG, resize, blur, and crop settings are deterministic for fixed library versions and inputs.

The paper-level robustness summary is provided in:

```text
results/robustness_results.csv
```

## Reproducibility notes

The released checkpoints and exact dataset manifest are the reference assets for evaluating the reported model. Retraining scripts reproduce the documented training protocol, but GPU/CUDA/cuDNN kernels and other implementation-level nondeterminism can cause small numerical differences across systems.

For dataset identity, use the SHA-256 hashes in `datasets/manifests/dataset_manifest.csv` rather than relying only on filenames or mutable upstream dataset revisions.

## Checkpoint and repository validation

Inspect checkpoint metadata:

```bash
python tools/inspect_checkpoints.py
```

Run the repository tests:

```bash
python -m pytest -q tests/test_repository_assets.py
```

## Third-party components

RIDER depends on third-party software and public datasets that remain subject to their respective licenses and terms. See [`THIRD_PARTY.md`](THIRD_PARTY.md) for details.
