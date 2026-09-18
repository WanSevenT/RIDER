# Dataset preparation

RIDER uses a unified benchmark assembled from multiple public sources. The original images are **not redistributed** in this repository. Obtain each source dataset according to its original license/terms and organize the retained samples using the released manifest.

## Exact split sizes

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 79,987 | 80,000 | 159,987 |
| Validation | 1,597 | 1,600 | 3,197 |
| Test | 60,677 | 60,707 | 121,384 |
| **Total** | **142,261** | **142,307** | **284,568** |

Training/validation contain five generator groups: LDM, ProGAN, SD1.4, SD2.1, and SDXL. The test split contains 20 datasets/generator groups, including FLUX, SD3.5, GPT-Image 1.5, and GPT-Image 2.0.

## Expected directory layout

```text
dataset/
├── train/
│   ├── ldm/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   └── progan/<optional-category>/{0_real,1_fake}/
├── val/
└── test/
```

The released `relative_path` field is relative to its split directory. For example, a manifest row with `split=train` and `relative_path=ldm/0_real/x.jpg` maps to `dataset/train/ldm/0_real/x.jpg`.

## Released manifests

- `dataset_manifest.csv`: exact retained samples, labels, relative paths, file sizes, and SHA-256 hashes.
- `dataset_counts.csv`: counts by split/source/model.
- `rider_dataset_counts_per_model.csv`: publication-facing counts by model.
- `rider_dataset_source_summary.csv`: source-level summary.
- `dataset_audit.json`: split audit with local machine paths removed.
- `dataset_manifest_template.csv`: schema template.

Verify file presence:

```bash
python tools/verify_dataset.py --root ./dataset
```

Verify all released SHA-256 hashes as well:

```bash
python tools/verify_dataset.py --root ./dataset --check_sha256
```

## Source datasets

The benchmark composition in the supplied manifest uses CNNDetection, Co-Spy-Bench, ComplexDataLab/OpenFake (GPT-Image 1.5), and the GPT-Image-2 Twitter collection. Consult the paper and each source project for their download instructions and redistribution terms.
