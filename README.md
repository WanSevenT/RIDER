# RIDER: Reliability-Informed Dual-Expert Routing for AI-Generated Image Detection

RIDER is a two-stage AI-generated image detector designed for cross-generator generalization and robustness to common image degradations. It independently learns complementary semantic and artifact experts, calibrates their predictions, and routes between them using input-dependent reliability cues.

This release is organized around the confirmed final checkpoint set corresponding to the paper result of **87.48% macro ACC / 95.26% macro AP** on the 20-dataset evaluation. The robustness protocol contains **11 perturbation settings** and reports **83.11% average ACC / 90.75% average AP**.

## Release contents

```text
RIDER-release/
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-optional.txt
├── configs/final_config.yaml
├── src/
│   ├── train_semantic.py
│   ├── train_artifact.py
│   ├── generate_clip_nn_scores.py
│   └── train_rider.py
├── rider/checkpoint.py
├── scripts/
│   ├── 01_train_semantic.sh
│   ├── 02_train_artifact.sh
│   ├── 03_generate_clip_nn.sh
│   ├── 04_train_rider.sh
│   └── 05_eval_rider.sh
├── datasets/
│   ├── README.md
│   └── manifests/
├── weights/
├── experiments/
│   ├── semantic_ablation/
│   └── artifact_ablation/
├── results/
│   ├── main_results.csv
│   ├── ablation_results.csv
│   └── robustness_results.csv
├── tools/
└── tests/
```

The `src/` files preserve the supplied experimental implementations. `src/train_rider.py` has only two release-compatibility additions: support for `semantic_best.pth`'s CLIP delta format and for metadata stored in the compact artifact checkpoint.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For CUDA, install the PyTorch build appropriate for your driver/CUDA environment if the generic pip wheel is not suitable. The final RIDER configuration does not require `diffusers`; legacy reconstruction-residual experiments can install `requirements-optional.txt`.

## Dataset

The original images are not redistributed. Prepare the public source datasets and reproduce the exact retained sample set using `datasets/manifests/dataset_manifest.csv`. See `datasets/README.md`.

Quick validation:

```bash
python tools/verify_dataset.py --root ./dataset
```

The exact released split contains 159,987 train, 3,197 validation, and 121,384 test images.

## Pretrained weights

Compact weights are included in `weights/` for this draft release. Their required loading order is:

```text
OpenAI CLIP ViT-L/14
        ↓
semantic_best.pth
        ↓
artifact_best.pth
        ↓
fusion_best.pth
```

The fusion checkpoint must be applied last because it contains the Phase-II-updated artifact BatchNorm buffers. Check file hashes with:

```bash
sha256sum -c weights/checksums.sha256
```

## Training

Run the four stages in order:

```bash
bash scripts/01_train_semantic.sh
bash scripts/02_train_artifact.sh
bash scripts/03_generate_clip_nn.sh
bash scripts/04_train_rider.sh
```

The final configuration uses:

- Semantic expert: CLIP ViT-L/14, 224×224, final visual Transformer block + LayerNorm, 5 epochs, patch shuffle disabled.
- Artifact expert: 224×224, spectral magnitude + Haar wavelet + MS-NPR, scales {0.25, 0.5, 0.75}, 20 epochs.
- CLIP-NN: k=10, at most 500 samples per generator/class, tau=40.
- Phase II: expert parameter gradients disabled; router trained up to 20 epochs; final selected checkpoint is epoch 4.

The artifact launch script explicitly uses `--normalized_artifact_inputs` to match the final RIDER path.

## Evaluation

Generate CLIP-NN scores for your prepared data first, then run:

```bash
bash scripts/03_generate_clip_nn.sh
bash scripts/05_eval_rider.sh
```

The evaluation script uses the fixed 0.5 operating point for the reported ACC protocol and writes generated JSON files under `outputs/eval/`. Generated outputs are ignored by Git so that `results/` remains a compact set of paper-level summaries.

## Results and ablations

To keep the repository close to the style of compact CVPR/ICML code releases, `results/` contains only three paper-level CSV files:

- `results/main_results.csv`: headline full-test, unseen-dataset, perturbation, and final expert results.
- `results/ablation_results.csv`: the ablation values reported in the current manuscript.
- `results/robustness_results.csv`: the robustness summary used by the paper.

Ablation launch commands and config snapshots live under `experiments/`, rather than `results/`. Raw logs, per-image predictions, epoch outputs, and earlier protocol snapshots are not committed.

## Check release integrity

```bash
python tools/inspect_release.py
python -m pytest tests/test_release_assets.py
```

## License

No software license is included in this draft package. Until a license is added, normal copyright rules apply; public users do not automatically receive permission to modify or redistribute the code. Before a public release, choose a license that is compatible with your institution/project policy and the third-party components you depend on.

## Citation

Citation information can be added after publication or de-anonymization.
