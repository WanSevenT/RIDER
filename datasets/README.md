# Dataset preparation

RIDER uses a unified benchmark assembled from public data sources. The original images are **not redistributed** in this repository. Obtain the upstream datasets under their original licenses/terms, construct the RIDER directory layout, and use the released manifest to verify exact membership.

## Exact split sizes

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 79,987 | 80,000 | 159,987 |
| Validation | 1,597 | 1,600 | 3,197 |
| Test | 60,677 | 60,707 | 121,384 |
| **Total** | **142,261** | **142,307** | **284,568** |

Training/validation contain five generator groups: LDM, ProGAN, SD1.4, SD2.1, and SDXL. The test split contains 20 dataset/model groups, including FLUX, SD3.5, GPT-Image 1.5, and GPT-Image 2.0.

## Upstream sources

| RIDER source/group | Upstream project | Use in RIDER | Public entry point |
|---|---|---|---|
| CNNDetection | CNNDetection | ProGAN training data and legacy GAN/image-to-image test groups | https://github.com/PeterWang512/CNNDetection |
| Co-Spy-Bench | CO-SPY / CO-SPY-Bench | LDM, SD1.4, SD2.1, SDXL, SD3.5, FLUX; CC3M/Flickr real-image pools | https://github.com/Megum1/CO-SPY |
| ComplexDataLab/OpenFake | OpenFake | GPT-Image 1.5 synthetic test images | https://huggingface.co/datasets/ComplexDataLab/OpenFake |
| GPT-Image-2 Twitter | GPT-Image-2 in the Wild | GPT-Image 2.0 synthetic test images | https://huggingface.co/datasets/Scam-AI/gpt-image-2 |

For the GPT-Image 1.5 and GPT-Image 2.0 benchmark groups, the synthetic images come from the corresponding upstream source. Real images are drawn from the CC3M/Flickr subsets distributed with CO-SPY, consistent with the manuscript protocol. The manifest `source` field identifies the RIDER benchmark group; it should not be interpreted as a per-file statement that every real image originated from the named synthetic-image dataset.

Upstream repositories and dataset cards may change over time. The SHA-256 values in the RIDER manifest are the authoritative identity check for the samples used in the reported experiments.

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

The released `relative_path` field is relative to its split directory. For example, a manifest row with `split=train` and `relative_path=ldm/0_real/x.jpg` maps to:

```text
dataset/train/ldm/0_real/x.jpg
```

## Released manifests

- `dataset_manifest.csv`: exact retained samples, labels, relative paths, file sizes, and SHA-256 hashes.
- `dataset_counts.csv`: counts by split/source/model.
- `rider_dataset_counts_per_model.csv`: publication-facing counts by model.
- `rider_dataset_source_summary.csv`: source-level summary.
- `dataset_audit.json`: split audit with local machine paths removed.
- `dataset_manifest_template.csv`: manifest schema template.

Verify file presence:

```bash
python tools/verify_dataset.py --root ./dataset
```

Verify all released SHA-256 hashes:

```bash
python tools/verify_dataset.py --root ./dataset --check_sha256
```

A successful SHA-256 check is the strongest confirmation that the locally prepared benchmark matches the released RIDER sample set.
