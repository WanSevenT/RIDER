#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure artifact-only trainer aligned with the v13 artifact_only path.

Kept from the original artifact_only path:
- recursive dataset scanning
- artifact light augmentation
- stage12 sampler + dynamic hard mining
- multiscale artifact extractor with optional reconstruction_residual
- ArtifactOnlyLoss
- optimizer using base_lr for artifact params
- fixed-0.5 validation artifact accuracy for best-checkpoint selection
- post-training fixed05 / val-selected / test-searched reports
- fake-only SupCon + fake-only domain adversarial auxiliary losses

Refined for standalone purity and stability:
- no semantic / CLIP / fusion module construction
- artifact branch initialization uses a dedicated RNG seed so removal of unrelated modules does not change artifact init
- auxiliary losses can be delayed and linearly ramped in after the main classifier starts to stabilize

Useful-branch version after 13-config ablation:
- keep only spectral_mag, wavelet, and npr as selectable/active artifact branches
- remove high_freq, compression, spectral_phase, and reconstruction_residual from default presets
- avoid VAE reconstruction_residual by default to reduce memory risk

B1 MS-NPR version:
- replace single-scale NPR with multi-scale NPR by default
- default NPR scales are 0.25, 0.50, and 0.75
- use --npr_scales 0.5 to reproduce the original single-scale NPR baseline
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

try:
    from sklearn.metrics import average_precision_score as sk_average_precision_score
except Exception:
    sk_average_precision_score = None

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBU = True
except Exception:
    HAS_ALBU = False

try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Final useful branches selected by ablation:
#   1) spectral_mag: useful frequency-magnitude cue
#   2) wavelet: strongest core local artifact cue
#   3) npr: strongest added residual cue
# Removed from the selectable set after ablation: high_freq, compression,
# spectral_phase, reconstruction_residual. The corresponding classes are kept
# below for compatibility with old checkpoints/code reading, but they are no
# longer enabled by default nor accepted by --artifact_branches.
DEFAULT_ARTIFACT_BRANCHES = ("spectral_mag", "wavelet", "npr")
ALL_ARTIFACT_BRANCHES = (
    "high_freq",
    "compression",
    "spectral_mag",
    "spectral_phase",
    "wavelet",
    "npr",
    "reconstruction_residual",
)
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def create_grad_scaler(enabled: bool):
    if not enabled:
        return None
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=True)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)
    return None


def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False


def seed_torch_only(seed: int):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_artifact_init_seed(global_seed: int, artifact_init_seed: int) -> int:
    artifact_init_seed = int(artifact_init_seed)
    return int(global_seed) if artifact_init_seed < 0 else artifact_init_seed


def compute_delayed_weight(epoch_idx: int, target_weight: float, start_epoch: int, ramp_epochs: int) -> float:
    target_weight = float(target_weight)
    if target_weight <= 0.0:
        return 0.0
    start_epoch = max(int(start_epoch), 0)
    ramp_epochs = int(ramp_epochs)
    if epoch_idx < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return target_weight
    progress = float(epoch_idx - start_epoch + 1) / float(ramp_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return target_weight * progress


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if device_arg in {"auto", "cuda"}:
        return "cuda:0"
    return device_arg


class ImageForgeryDataset(Dataset):
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root_dir: str, transform=None, verify_images: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.verify_images = verify_images
        self.samples: List[Dict[str, object]] = []
        self.images: List[str] = []
        self.labels: List[int] = []
        self.categories: List[str] = []
        self.models: List[str] = []
        self.subsets: List[str] = []
        self.skipped_corrupt: List[str] = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root not found: {root_dir}")
        self._scan()
        self.num_skipped_corrupt = len(self.skipped_corrupt)
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found under: {root_dir}. Expected folders like "
                f"model/0_real, model/1_fake, or model/category/0_real, model/category/1_fake."
            )

    def _scan(self):
        for current_root, dirnames, _ in os.walk(self.root_dir, topdown=True):
            dirnames.sort()
            has_real = "0_real" in dirnames
            has_fake = "1_fake" in dirnames
            if not (has_real or has_fake):
                continue
            rel_dir = os.path.relpath(current_root, self.root_dir)
            rel_dir = "" if rel_dir == "." else rel_dir.replace("\\", "/")
            category = rel_dir if rel_dir else Path(current_root).name
            model_name, subset_name = self._split_category(category)
            if has_real:
                self._ingest_label_dir(os.path.join(current_root, "0_real"), 0, category, model_name, subset_name)
            if has_fake:
                self._ingest_label_dir(os.path.join(current_root, "1_fake"), 1, category, model_name, subset_name)
            dirnames[:] = [d for d in dirnames if d not in {"0_real", "1_fake"}]

    @staticmethod
    def _split_category(category: str) -> Tuple[str, str]:
        category = str(category).replace("\\", "/").strip("/")
        parts = [p for p in category.split("/") if p]
        if not parts:
            return "unknown", ""
        return parts[0], "/".join(parts[1:]) if len(parts) > 1 else ""

    def _is_valid_image(self, path: str) -> bool:
        if not self.verify_images:
            return True
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.convert("RGB")
            return True
        except Exception:
            return False

    def _ingest_label_dir(self, path: str, label: int, category: str, model_name: str, subset_name: str):
        if not os.path.isdir(path):
            return
        for img_name in sorted(os.listdir(path)):
            if not img_name.lower().endswith(self.IMG_EXTS):
                continue
            img_path = os.path.join(path, img_name)
            if not self._is_valid_image(img_path):
                self.skipped_corrupt.append(img_path)
                continue
            self.samples.append({
                "path": img_path,
                "label": int(label),
                "category": category,
                "model": model_name,
                "subset": subset_name,
            })
            self.images.append(img_path)
            self.labels.append(int(label))
            self.categories.append(category)
            self.models.append(model_name)
            self.subsets.append(subset_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = str(sample["path"])
        label = int(sample["label"])
        category = str(sample["category"])
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            if self.transform is None:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                out = self.transform(image=image)
                image = out["image"] if isinstance(out, dict) and "image" in out else out
            return image, label, category
        except Exception:
            return None


class MetadataSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)
        self.samples = [dataset.samples[i] for i in self.indices] if hasattr(dataset, "samples") else None
        self.images = [dataset.images[i] for i in self.indices] if hasattr(dataset, "images") else []
        self.labels = [dataset.labels[i] for i in self.indices] if hasattr(dataset, "labels") else []
        self.categories = [dataset.categories[i] for i in self.indices] if hasattr(dataset, "categories") else []
        self.models = [dataset.models[i] for i in self.indices] if hasattr(dataset, "models") else []
        self.subsets = [dataset.subsets[i] for i in self.indices] if hasattr(dataset, "subsets") else []
        self.root_dir = getattr(dataset, "root_dir", "")
        self.transform = getattr(dataset, "transform", None)
        self.verify_images = getattr(dataset, "verify_images", False)
        self.num_skipped_corrupt = 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def forgiving_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def extract_group_key(category: str) -> str:
    c = (category or "").lower()
    if "_" in c:
        return c.split("_")[-1]
    return c


def validate_image_size(image_size: int) -> int:
    image_size = int(image_size)
    if image_size <= 0:
        raise ValueError("--image_size must be positive")
    if image_size % 32 != 0:
        raise ValueError("--image_size must be divisible by 32 so branch feature maps stay aligned")
    return image_size


def get_artifact_train_transform(image_size: int = 256):
    image_size = validate_image_size(image_size)
    if HAS_ALBU:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_range=(75, 100), p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.04, contrast_limit=0.04, p=0.12),
            A.Normalize(mean=list(CLIP_IMAGE_MEAN), std=list(CLIP_IMAGE_STD)),
            ToTensorV2(),
        ])

    class _Simple:
        def __call__(self, image):
            img = Image.fromarray(image).resize((image_size, image_size))
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = (arr - np.array(CLIP_IMAGE_MEAN, dtype=np.float32)) / np.array(CLIP_IMAGE_STD, dtype=np.float32)
            return {"image": torch.from_numpy(arr).permute(2, 0, 1)}

    return _Simple()


def get_val_transform(image_size: int = 256):
    image_size = validate_image_size(image_size)
    if HAS_ALBU:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=list(CLIP_IMAGE_MEAN), std=list(CLIP_IMAGE_STD)),
            ToTensorV2(),
        ])

    class _Simple:
        def __call__(self, image):
            img = Image.fromarray(image).resize((image_size, image_size))
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = (arr - np.array(CLIP_IMAGE_MEAN, dtype=np.float32)) / np.array(CLIP_IMAGE_STD, dtype=np.float32)
            return {"image": torch.from_numpy(arr).permute(2, 0, 1)}

    return _Simple()


def compute_class_weights_from_labels(labels: List[int]) -> torch.Tensor:
    t = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(t, minlength=2).float().clamp(min=1.0)
    return 1.0 / counts


def build_stage12_sampler(dataset: ImageForgeryDataset, balance_labels: bool = True):
    labels = np.array(dataset.labels, dtype=np.int64)
    models = np.array(dataset.models)
    cats = np.array(dataset.categories)
    group_keys = np.array([extract_group_key(c) for c in cats])

    pair_counts = Counter(zip(models.tolist(), labels.tolist()))
    weights = np.array(
        [1.0 / float(pair_counts[(m, int(y))]) for m, y in zip(models, labels)],
        dtype=np.float64,
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    sampler.base_weights = weights.copy()

    group_counts = Counter(group_keys.tolist())
    label_counts = Counter(labels.tolist())
    return sampler, weights, dict(group_counts), dict(label_counts)


def update_sampler_for_hard_mining(dataset: ImageForgeryDataset, sampler: WeightedRandomSampler, category_error_stats: Dict[str, Dict[str, float]], dynamic_boost: float = 2.0):
    base_weights = np.array(getattr(sampler, "base_weights", np.ones(len(dataset))), dtype=np.float64)
    new_weights = base_weights.copy()
    for i, (cat, label) in enumerate(zip(dataset.categories, dataset.labels)):
        stat = category_error_stats.get(str(cat).lower())
        if stat is None:
            continue
        err_rate = float(stat.get("err_rate", 0.0))
        fake_err_rate = float(stat.get("fake_err_rate", err_rate))
        boost = 1.0
        if int(label) == 1:
            boost += dynamic_boost * fake_err_rate
        else:
            boost += 0.25 * dynamic_boost * err_rate
        new_weights[i] *= boost
    sampler.weights = torch.as_tensor(new_weights, dtype=torch.double)


def _clip_denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(CLIP_IMAGE_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(CLIP_IMAGE_STD).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def normalize_npr_scales(scales: Optional[Sequence[float]] = None) -> Tuple[float, ...]:
    if scales is None:
        scales = (0.25, 0.5, 0.75)
    out: List[float] = []
    for s in scales:
        v = float(s)
        if not (0.0 < v < 1.0):
            raise ValueError(f"NPR scale must be in (0, 1), got {v}")
        if v not in out:
            out.append(v)
    if len(out) == 0:
        raise ValueError("At least one NPR scale is required")
    return tuple(out)


class FixedLatentAutoencoderReconstructor(nn.Module):
    def __init__(self, vae_path: str, subfolder: Optional[str] = None, torch_dtype: str = "auto"):
        super().__init__()
        if AutoencoderKL is None:
            raise ImportError("reconstruction_residual branch requires diffusers")
        load_kwargs = {}
        if subfolder:
            load_kwargs["subfolder"] = subfolder
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        requested_dtype = None
        if torch_dtype != "auto":
            requested_dtype = dtype_map.get(torch_dtype.lower(), torch.float32)
            load_kwargs["torch_dtype"] = requested_dtype
        self.vae = AutoencoderKL.from_pretrained(vae_path, **load_kwargs)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        if requested_dtype is not None:
            self.vae_runtime_dtype = requested_dtype
        else:
            first_param = next(iter(self.vae.parameters()), None)
            self.vae_runtime_dtype = first_param.dtype if first_param is not None else torch.float32

    @torch.no_grad()
    def reconstruct(self, x_01: torch.Tensor) -> torch.Tensor:
        x = (x_01 * 2.0 - 1.0).clamp(-1.0, 1.0).to(dtype=self.vae_runtime_dtype)
        enc_out = self.vae.encode(x)
        latent_dist = getattr(enc_out, "latent_dist", None)
        if latent_dist is None:
            raise RuntimeError("Unexpected VAE encode output: missing latent_dist")
        z = latent_dist.mode()
        scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
        dec_out = self.vae.decode(z * scaling_factor / scaling_factor)
        sample = getattr(dec_out, "sample", dec_out[0] if isinstance(dec_out, (tuple, list)) else dec_out)
        return ((sample.float() + 1.0) * 0.5).clamp(0.0, 1.0)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        p = k // 2 if p is None else p
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FFTMagnitudePhase(nn.Module):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        else:
            gray = x[:, 0]
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1e-8)
        phase = torch.angle(fft_shift)
        b = magnitude.shape[0]
        magnitude = (magnitude - magnitude.view(b, -1).mean(dim=1, keepdim=True).view(b, 1, 1)) / (
            magnitude.view(b, -1).std(dim=1, keepdim=True).view(b, 1, 1) + 1e-6
        )
        phase = phase / math.pi
        return magnitude.unsqueeze(1), phase.unsqueeze(1)


class SpectralBranch(nn.Module):
    def __init__(self, in_ch: int = 1, mid_ch: int = 48, token_dim: int = 128):
        super().__init__()
        self.stem = ConvBNAct(in_ch, mid_ch, 3, 1)
        self.down1 = ConvBNAct(mid_ch, mid_ch * 2, 3, 2)
        self.down2 = ConvBNAct(mid_ch * 2, mid_ch * 2, 3, 2)
        self.out28 = ConvBNAct(mid_ch * 2, token_dim, 3, 2)
        self.out14 = ConvBNAct(token_dim, token_dim, 3, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        feat28 = self.out28(x)
        feat14 = self.out14(feat28)
        return feat28, feat14


class SpatialBranch(nn.Module):
    def __init__(self, in_ch: int, chs=(32, 64, 96, 128)):
        super().__init__()
        c1, c2, c3, c4 = chs
        self.stem = ConvBNAct(in_ch, c1, 3, 1)
        self.down1 = ConvBNAct(c1, c2, 3, 2)
        self.down2 = ConvBNAct(c2, c3, 3, 2)
        self.out28 = ConvBNAct(c3, c4, 3, 2)
        self.out14 = ConvBNAct(c4, c4, 3, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        feat28 = self.out28(x)
        feat14 = self.out14(feat28)
        return feat28, feat14


class CompressionBranch(nn.Module):
    def __init__(self, token_dim: int = 128):
        super().__init__()
        self.block_embed = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.mid = ConvBNAct(32, 64, 3, 1)
        self.out28 = ConvBNAct(64, token_dim, 3, 2)
        self.out14 = ConvBNAct(token_dim, token_dim, 3, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.block_embed(x)
        x = self.mid(x)
        feat28 = self.out28(x)
        feat14 = self.out14(feat28)
        return feat28, feat14


class HaarWaveletBranch(nn.Module):
    def __init__(self, token_dim: int = 128):
        super().__init__()
        haar = torch.tensor([
            [[1, 1], [-1, -1]],
            [[1, -1], [1, -1]],
            [[1, -1], [-1, 1]],
        ], dtype=torch.float32) / 2.0
        self.register_buffer("haar_kernels", haar.unsqueeze(1), persistent=False)
        self.proj = nn.Sequential(
            ConvBNAct(3, 32, 3, 1),
            ConvBNAct(32, 64, 3, 2),
            ConvBNAct(64, 96, 3, 1),
        )
        self.out28 = ConvBNAct(96, token_dim, 3, 2)
        self.out14 = ConvBNAct(token_dim, token_dim, 3, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        else:
            gray = x[:, 0]
        gray = gray.unsqueeze(1)
        coeff = F.conv2d(gray, self.haar_kernels.to(device=gray.device, dtype=gray.dtype), stride=2, padding=0).abs()
        mid = self.proj(coeff)
        feat28 = self.out28(mid)
        feat14 = self.out14(feat28)
        return feat28, feat14, coeff


class NPRResidualBranch(nn.Module):
    def __init__(
        self,
        token_dim: int = 128,
        downsample_scale: float = 0.5,
        downsample_scales: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if downsample_scales is None:
            downsample_scales = (downsample_scale,)
        self.downsample_scales = normalize_npr_scales(downsample_scales)
        # Each NPR scale contributes residual and |residual|, each with 3 RGB channels.
        self.branch = SpatialBranch(6 * len(self.downsample_scales), chs=(32, 64, 96, token_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x01 = _clip_denormalize(x)
        h, w = x01.shape[-2:]
        residuals: List[torch.Tensor] = []
        npr_parts: List[torch.Tensor] = []
        for scale in self.downsample_scales:
            low = F.interpolate(
                x01,
                scale_factor=float(scale),
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            up = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
            residual = x01 - up
            residuals.append(residual)
            npr_parts.extend([residual, residual.abs()])
        npr_input = torch.cat(npr_parts, dim=1)
        feat28, feat14 = self.branch(npr_input)
        residual_maps = residuals[0] if len(residuals) == 1 else torch.cat(residuals, dim=1)
        return feat28, feat14, residual_maps


class MultiScaleLocalArtifactExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int = 192,
        token_dim: int = 128,
        token_pool_size: int = 14,
        active_branches: Optional[Sequence[str]] = None,
        reconstruction_vae_path: Optional[str] = None,
        reconstruction_vae_subfolder: Optional[str] = None,
        reconstruction_vae_dtype: str = "auto",
        reconstruction_use_fft_residual: bool = False,
        npr_scales: Optional[Sequence[float]] = None,
        raw_artifact_inputs: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.token_pool_size = token_pool_size
        self.reconstruction_use_fft_residual = reconstruction_use_fft_residual
        self.npr_scales = normalize_npr_scales(npr_scales)
        self.raw_artifact_inputs = bool(raw_artifact_inputs)
        requested = tuple(active_branches) if active_branches is not None else DEFAULT_ARTIFACT_BRANCHES
        invalid = sorted(set(requested) - set(ALL_ARTIFACT_BRANCHES))
        if invalid:
            raise ValueError(f"Unknown artifact branches: {invalid}")
        self.branch_names = [name for name in ALL_ARTIFACT_BRANCHES if name in requested]
        if not self.branch_names:
            raise ValueError("At least one artifact branch must be enabled")

        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer("laplacian_kernel", lap.view(1, 1, 3, 3).repeat(3, 1, 1, 1), persistent=False)
        self.fft = FFTMagnitudePhase() if any(name in self.branch_names for name in ("spectral_mag", "spectral_phase", "reconstruction_residual")) else None

        self.high_freq_branch = SpatialBranch(3, chs=(32, 64, 96, token_dim)) if "high_freq" in self.branch_names else None
        self.compression_branch = CompressionBranch(token_dim=token_dim) if "compression" in self.branch_names else None
        self.spectral_mag_branch = SpectralBranch(in_ch=1, mid_ch=48, token_dim=token_dim) if "spectral_mag" in self.branch_names else None
        self.spectral_phase_branch = SpectralBranch(in_ch=1, mid_ch=48, token_dim=token_dim) if "spectral_phase" in self.branch_names else None
        self.wavelet_branch = HaarWaveletBranch(token_dim=token_dim) if "wavelet" in self.branch_names else None
        self.npr_branch = NPRResidualBranch(token_dim=token_dim, downsample_scales=self.npr_scales) if "npr" in self.branch_names else None
        self.reconstructor = None
        self.reconstruction_branch = None
        if "reconstruction_residual" in self.branch_names:
            if not reconstruction_vae_path:
                raise ValueError("reconstruction_residual requires --reconstruction_vae_path")
            recon_in_ch = 10 if reconstruction_use_fft_residual else 9
            self.reconstructor = FixedLatentAutoencoderReconstructor(
                vae_path=reconstruction_vae_path,
                subfolder=reconstruction_vae_subfolder,
                torch_dtype=reconstruction_vae_dtype,
            )
            self.reconstruction_branch = SpatialBranch(recon_in_ch, chs=(32, 64, 96, token_dim))

        self.branch_attention = nn.Sequential(
            nn.Linear(len(self.branch_names) * token_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(self.branch_names)),
            nn.Softmax(dim=1),
        )
        self.level28_refine = nn.Sequential(ConvBNAct(token_dim, token_dim, 3, 1), ConvBNAct(token_dim, token_dim, 3, 1))
        self.level14_refine = nn.Sequential(ConvBNAct(token_dim, token_dim, 3, 1), ConvBNAct(token_dim, token_dim, 3, 1))
        self.feature_fusion = nn.Sequential(
            nn.Linear(token_dim * 4, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, feature_dim),
        )

    def _weighted_sum(self, maps: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        fused = 0.0
        for i, m in enumerate(maps):
            fused = fused + weights[:, i].view(-1, 1, 1, 1) * m
        return fused

    def _make_tokens(self, feat28: torch.Tensor, feat14: torch.Tensor) -> torch.Tensor:
        pooled28 = F.adaptive_avg_pool2d(feat28, (self.token_pool_size, self.token_pool_size))
        pooled14 = F.adaptive_avg_pool2d(feat14, (self.token_pool_size, self.token_pool_size))
        tokens28 = pooled28.flatten(2).transpose(1, 2)
        tokens14 = pooled14.flatten(2).transpose(1, 2)
        return torch.cat([tokens28, tokens14], dim=1)

    def _build_reconstruction_residual_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.reconstructor is None:
            raise RuntimeError("Reconstruction branch requested but reconstructor is missing")
        x_img = _clip_denormalize(x)
        x_hat = self.reconstructor.reconstruct(x_img)
        lap_k = self.laplacian_kernel.to(device=x_img.device, dtype=x_img.dtype)
        lap_x = F.conv2d(x_img, lap_k, padding=1, groups=3)
        lap_hat = F.conv2d(x_hat, lap_k, padding=1, groups=3)
        recon_parts = [x_img - x_hat, torch.abs(x_img - x_hat), lap_x - lap_hat]
        if self.reconstruction_use_fft_residual:
            if self.fft is None:
                raise RuntimeError("FFT helper is unavailable")
            mag_x, _ = self.fft(x_img)
            mag_hat, _ = self.fft(x_hat)
            recon_parts.append(mag_x - mag_hat)
        return torch.cat(recon_parts, dim=1)

    def forward(self, x: torch.Tensor, return_maps: bool = False):
        # Artifact cues should be computed in the raw [0, 1] image domain.
        # The classifier still receives normalized tensors, but FFT / wavelet / Laplacian
        # are sensitive to per-channel CLIP normalization and mean/std scaling.
        x_artifact = _clip_denormalize(x) if self.raw_artifact_inputs else x
        lap_k = self.laplacian_kernel.to(device=x_artifact.device, dtype=x_artifact.dtype)
        residuals = F.conv2d(x_artifact, lap_k, padding=1, groups=3)
        branch_maps28: List[torch.Tensor] = []
        branch_maps14: List[torch.Tensor] = []
        pooled_parts: List[torch.Tensor] = []
        maps: Dict[str, torch.Tensor] = {"residuals": residuals}

        if self.high_freq_branch is not None:
            hf28, hf14 = self.high_freq_branch(residuals)
            branch_maps28.append(hf28)
            branch_maps14.append(hf14)
            pooled_parts.append(F.adaptive_avg_pool2d(hf14, 1).flatten(1))
            maps.update({"high_freq_map_28": hf28, "high_freq_map_14": hf14})

        if self.compression_branch is not None:
            comp28, comp14 = self.compression_branch(x)
            branch_maps28.append(comp28)
            branch_maps14.append(comp14)
            pooled_parts.append(F.adaptive_avg_pool2d(comp14, 1).flatten(1))
            maps.update({"compression_map_28": comp28, "compression_map_14": comp14})

        mag = phase = None
        if self.fft is not None:
            mag, phase = self.fft(x_artifact)
            maps.update({"spectral_magnitude": mag, "spectral_phase": phase})

        if self.spectral_mag_branch is not None:
            sm28, sm14 = self.spectral_mag_branch(mag)
            branch_maps28.append(sm28)
            branch_maps14.append(sm14)
            pooled_parts.append(F.adaptive_avg_pool2d(sm14, 1).flatten(1))
            maps.update({"spectral_mag_map_28": sm28, "spectral_mag_map_14": sm14})

        if self.spectral_phase_branch is not None:
            sp28, sp14 = self.spectral_phase_branch(phase)
            branch_maps28.append(sp28)
            branch_maps14.append(sp14)
            pooled_parts.append(F.adaptive_avg_pool2d(sp14, 1).flatten(1))
            maps.update({"spectral_phase_map_28": sp28, "spectral_phase_map_14": sp14})

        if self.wavelet_branch is not None:
            wav28, wav14, wavelet_coeffs = self.wavelet_branch(x_artifact)
            branch_maps28.append(wav28)
            branch_maps14.append(wav14)
            pooled_parts.append(F.adaptive_avg_pool2d(wav14, 1).flatten(1))
            maps.update({"wavelet_coeffs": wavelet_coeffs, "wavelet_map_28": wav28, "wavelet_map_14": wav14})

        if self.npr_branch is not None:
            npr28, npr14, npr_residual = self.npr_branch(x)
            branch_maps28.append(npr28)
            branch_maps14.append(npr14)
            pooled_parts.append(F.adaptive_avg_pool2d(npr14, 1).flatten(1))
            maps.update({"npr_residual": npr_residual, "npr_map_28": npr28, "npr_map_14": npr14})

        if self.reconstruction_branch is not None:
            recon_input = self._build_reconstruction_residual_input(x)
            rec28, rec14 = self.reconstruction_branch(recon_input)
            branch_maps28.append(rec28)
            branch_maps14.append(rec14)
            pooled_parts.append(F.adaptive_avg_pool2d(rec14, 1).flatten(1))
            maps.update({"reconstruction_map_28": rec28, "reconstruction_map_14": rec14})

        pooled = torch.cat(pooled_parts, dim=1)
        attention_weights = self.branch_attention(pooled)
        fused28 = self.level28_refine(self._weighted_sum(branch_maps28, attention_weights))
        fused14 = self.level14_refine(self._weighted_sum(branch_maps14, attention_weights))
        artifact_tokens = self._make_tokens(fused28, fused14)
        pooled28 = F.adaptive_avg_pool2d(fused28, 1).flatten(1)
        pooled14 = F.adaptive_avg_pool2d(fused14, 1).flatten(1)
        max28 = F.adaptive_max_pool2d(fused28, 1).flatten(1)
        max14 = F.adaptive_max_pool2d(fused14, 1).flatten(1)
        artifact_features = self.feature_fusion(torch.cat([pooled28, pooled14, max28, max14], dim=1))
        if not return_maps:
            return artifact_features, artifact_tokens, attention_weights
        maps.update({"artifact_fused_28": fused28, "artifact_fused_14": fused14})
        return artifact_features, artifact_tokens, attention_weights, maps


class ArtifactOnlyBranchDetector(nn.Module):
    def __init__(
        self,
        artifact_feature_dim: int = 192,
        artifact_token_dim: int = 128,
        token_pool_size: int = 14,
        artifact_feature_dropout: float = 0.15,
        artifact_branches: Optional[Sequence[str]] = None,
        reconstruction_vae_path: Optional[str] = None,
        reconstruction_vae_subfolder: Optional[str] = None,
        reconstruction_vae_dtype: str = "auto",
        reconstruction_use_fft_residual: bool = False,
        npr_scales: Optional[Sequence[float]] = None,
        raw_artifact_inputs: bool = True,
        artifact_aux_proj_hidden_dim: int = 192,
        artifact_aux_proj_dim: int = 128,
        artifact_aux_dropout: float = 0.05,
    ):
        super().__init__()
        self.artifact_feature_dim = int(artifact_feature_dim)
        self.artifact_feature_dropout = float(artifact_feature_dropout)
        self.artifact_branches = tuple(artifact_branches) if artifact_branches is not None else DEFAULT_ARTIFACT_BRANCHES
        self.reconstruction_use_fft_residual = bool(reconstruction_use_fft_residual)
        self.npr_scales = normalize_npr_scales(npr_scales)
        self.raw_artifact_inputs = bool(raw_artifact_inputs)
        self.artifact_aux_proj_hidden_dim = int(artifact_aux_proj_hidden_dim)
        self.artifact_aux_proj_dim = int(artifact_aux_proj_dim)
        self.artifact_aux_dropout = float(artifact_aux_dropout)
        self.artifact_extractor = MultiScaleLocalArtifactExtractor(
            feature_dim=self.artifact_feature_dim,
            token_dim=int(artifact_token_dim),
            token_pool_size=int(token_pool_size),
            active_branches=self.artifact_branches,
            reconstruction_vae_path=reconstruction_vae_path,
            reconstruction_vae_subfolder=reconstruction_vae_subfolder,
            reconstruction_vae_dtype=reconstruction_vae_dtype,
            reconstruction_use_fft_residual=self.reconstruction_use_fft_residual,
            npr_scales=self.npr_scales,
            raw_artifact_inputs=raw_artifact_inputs,
        )
        self.artifact_classifier = nn.Linear(self.artifact_feature_dim, 1)
        self.artifact_aux_projector = nn.Sequential(
            nn.LayerNorm(self.artifact_feature_dim),
            nn.Linear(self.artifact_feature_dim, self.artifact_aux_proj_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.artifact_aux_dropout),
            nn.Linear(self.artifact_aux_proj_hidden_dim, self.artifact_aux_proj_dim),
        )
        self.artifact_domain_classifier = None
        self.artifact_domain_num_domains = 0
        self.artifact_domain_hidden_dim = 0

    def configure_artifact_domain_classifier(self, num_domains: int, hidden_dim: int = 128):
        num_domains = int(num_domains or 0)
        hidden_dim = int(hidden_dim)
        if num_domains <= 1:
            self.artifact_domain_classifier = None
            self.artifact_domain_num_domains = 0
            self.artifact_domain_hidden_dim = 0
            return
        self.artifact_domain_classifier = nn.Sequential(
            nn.LayerNorm(self.artifact_aux_proj_dim),
            nn.Linear(self.artifact_aux_proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, num_domains),
        )
        self.artifact_domain_num_domains = num_domains
        self.artifact_domain_hidden_dim = hidden_dim

    def forward(self, x, return_details: bool = False, return_maps: bool = False):
        if return_maps:
            artifact_backbone_features, artifact_tokens, artifact_attention, artifact_maps = self.artifact_extractor(x, return_maps=True)
        else:
            artifact_backbone_features, artifact_tokens, artifact_attention = self.artifact_extractor(x, return_maps=False)
            artifact_maps = None
        classifier_features = artifact_backbone_features
        if self.training and self.artifact_feature_dropout > 0:
            classifier_features = F.dropout(classifier_features, p=self.artifact_feature_dropout, training=True)
        artifact_output = self.artifact_classifier(classifier_features)
        artifact_aux_features = F.normalize(self.artifact_aux_projector(artifact_backbone_features), dim=1)
        if not return_details:
            return artifact_output
        outputs = {
            "artifact_output": artifact_output,
            "artifact_features": classifier_features,
            "artifact_backbone_features": artifact_backbone_features,
            "artifact_aux_features": artifact_aux_features,
            "artifact_tokens": artifact_tokens,
            "artifact_attention": artifact_attention,
        }
        if artifact_maps is not None:
            outputs.update(artifact_maps)
        return outputs


class WeightedFocalBCE(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0, alpha: Optional[float] = 0.65):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor([1.0]))
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        pos_weight = self.pos_weight.to(logits.device, logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - pt).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal = focal * alpha_t
        return (bce * focal).mean()


class ArtifactOnlyLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, pos_scale: float = 1.5, focal_gamma: float = 2.0, focal_alpha: float = 0.65):
        super().__init__()
        if class_weights is not None and class_weights.numel() >= 2:
            pos_weight = (class_weights[1] / class_weights[0]).clamp(min=1e-6) * pos_scale
        else:
            pos_weight = torch.tensor([pos_scale], dtype=torch.float32)
        self.loss_fn = WeightedFocalBCE(pos_weight=pos_weight.float(), gamma=focal_gamma, alpha=focal_alpha)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor):
        targets = targets.float()
        artifact_loss = self.loss_fn(outputs["artifact_output"], targets)
        info = {
            "total_loss": float(artifact_loss.item()),
            "artifact_loss": float(artifact_loss.item()),
        }
        return artifact_loss, info


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambd)


def categories_to_model_names(categories: Sequence[object]) -> List[str]:
    return [str(v).replace("\\", "/").split("/")[0] for v in categories]


def encode_domain_labels(categories: Sequence[object], domain_to_idx: Dict[str, int], device: str) -> torch.Tensor:
    models = categories_to_model_names(categories)
    labels = [int(domain_to_idx.get(m, -1)) for m in models]
    return torch.tensor(labels, dtype=torch.long, device=device)


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if features is None or features.ndim != 2 or features.shape[0] < 2:
        return features.new_tensor(0.0) if isinstance(features, torch.Tensor) else torch.tensor(0.0)
    features = F.normalize(features, dim=1)
    labels = labels.view(-1)
    logits = torch.matmul(features, features.t()) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(logits) - torch.eye(logits.shape[0], device=logits.device, dtype=logits.dtype)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float() * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.new_tensor(0.0)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / positive_count[valid]
    return -mean_log_prob_pos.mean()


def compute_artifact_aux_losses(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    categories: Sequence[object],
    model: nn.Module,
    artifact_supcon_weight: float = 0.0,
    artifact_supcon_temperature: float = 0.07,
    artifact_domain_adv_weight: float = 0.0,
    artifact_domain_adv_lambda: float = 1.0,
    artifact_domain_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    base_model = unwrap_model(model)
    features = outputs.get("artifact_aux_features", outputs.get("artifact_features", None))
    total = labels.new_tensor(0.0)
    info = {"supcon_loss": 0.0, "domain_loss": 0.0}

    if artifact_supcon_weight > 0.0 and isinstance(features, torch.Tensor):
        fake_mask = labels.view(-1) > 0.5
        if int(fake_mask.sum().item()) >= 2:
            fake_features = features[fake_mask]
            fake_labels = torch.zeros(fake_features.shape[0], dtype=torch.long, device=fake_features.device)
            supcon = supervised_contrastive_loss(fake_features, fake_labels, temperature=artifact_supcon_temperature)
            total = total + artifact_supcon_weight * supcon
            info["supcon_loss"] = float(supcon.item())

    domain_classifier = getattr(base_model, "artifact_domain_classifier", None)
    if artifact_domain_adv_weight > 0.0 and artifact_domain_to_idx and domain_classifier is not None and isinstance(features, torch.Tensor):
        domain_labels = encode_domain_labels(categories, artifact_domain_to_idx, features.device)
        fake_mask = (labels.view(-1) > 0.5) & (domain_labels >= 0)
        if int(fake_mask.sum().item()) >= 2:
            fake_features = features[fake_mask]
            fake_domain_labels = domain_labels[fake_mask]
            reversed_features = grad_reverse(fake_features, artifact_domain_adv_lambda)
            domain_logits = domain_classifier(reversed_features)
            domain_loss = F.cross_entropy(domain_logits, fake_domain_labels)
            total = total + artifact_domain_adv_weight * domain_loss
            info["domain_loss"] = float(domain_loss.item())

    return total, info


def get_artifact_domain_to_idx(dataset: Dataset) -> Dict[str, int]:
    models = sorted({str(m) for m in getattr(dataset, "models", [])})
    return {m: i for i, m in enumerate(models)}


ABLATION_PRESET_TO_BRANCHES = {
    "useful3": ["spectral_mag", "wavelet", "npr"],
    "all_old_no_recon": [
        "high_freq",
        "compression",
        "spectral_mag",
        "spectral_phase",
        "wavelet",
        "npr",
    ],
    "all_old_with_recon": [
        "high_freq",
        "compression",
        "spectral_mag",
        "spectral_phase",
        "wavelet",
        "npr",
        "reconstruction_residual",
    ],
}


def resolve_artifact_branches_for_run(args) -> Tuple[List[str], bool]:
    if args.artifact_branches is not None:
        branches = list(args.artifact_branches)
    else:
        preset = str(getattr(args, "ablation_preset", "useful3") or "useful3").lower()
        if preset not in ABLATION_PRESET_TO_BRANCHES:
            raise ValueError(f"Unknown --ablation_preset: {preset}")
        branches = list(ABLATION_PRESET_TO_BRANCHES[preset])

    invalid = sorted(set(branches) - set(ALL_ARTIFACT_BRANCHES))
    if invalid:
        raise ValueError(
            f"Unsupported artifact branches in useful3 version: {invalid}. "
            f"Allowed branches: {list(ALL_ARTIFACT_BRANCHES)}"
        )

    reconstruction_use_fft_residual = bool(
        ("reconstruction_residual" in branches)
        and bool(getattr(args, "reconstruction_use_fft_residual", False))
    )
    return branches, reconstruction_use_fft_residual


def build_pure_artifact_model(
    args,
    branches: Sequence[str],
    reconstruction_use_fft_residual: bool,
    artifact_domain_to_idx: Optional[Dict[str, int]],
) -> ArtifactOnlyBranchDetector:
    artifact_init_seed = resolve_artifact_init_seed(args.seed, args.artifact_init_seed)
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        seed_torch_only(artifact_init_seed)
        model = ArtifactOnlyBranchDetector(
            artifact_feature_dim=args.artifact_feature_dim,
            artifact_token_dim=args.artifact_token_dim,
            token_pool_size=args.token_pool_size,
            artifact_feature_dropout=args.artifact_feature_dropout,
            artifact_branches=branches,
            reconstruction_vae_path=args.reconstruction_vae_path or None,
            reconstruction_vae_subfolder=args.reconstruction_vae_subfolder or None,
            reconstruction_vae_dtype=args.reconstruction_vae_dtype,
            reconstruction_use_fft_residual=reconstruction_use_fft_residual,
            npr_scales=args.npr_scales,
            raw_artifact_inputs=bool(args.raw_artifact_inputs),
            artifact_aux_proj_hidden_dim=args.artifact_aux_proj_hidden_dim,
            artifact_aux_proj_dim=args.artifact_aux_proj_dim,
            artifact_aux_dropout=args.artifact_aux_dropout,
        )
        if float(args.artifact_domain_adv_weight) > 0.0 and artifact_domain_to_idx and len(artifact_domain_to_idx) > 1:
            seed_torch_only(artifact_init_seed + 1)
            model.configure_artifact_domain_classifier(
                num_domains=len(artifact_domain_to_idx),
                hidden_dim=int(args.artifact_domain_hidden_dim),
            )
    return model


def freeze_for_artifact_only(model: ArtifactOnlyBranchDetector):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.artifact_extractor.parameters():
        p.requires_grad = True
    for p in model.artifact_classifier.parameters():
        p.requires_grad = True
    for p in model.artifact_aux_projector.parameters():
        p.requires_grad = True
    if getattr(model, "artifact_domain_classifier", None) is not None:
        for p in model.artifact_domain_classifier.parameters():
            p.requires_grad = True


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def build_optimizer(
    model: ArtifactOnlyBranchDetector,
    base_lr: float = 1e-4,
    artifact_aux_lr_scale: float = 0.5,
    artifact_domain_lr_scale: float = 0.25,
):
    main_params = [p for p in model.artifact_extractor.parameters() if p.requires_grad] + [p for p in model.artifact_classifier.parameters() if p.requires_grad]
    aux_params = [p for p in model.artifact_aux_projector.parameters() if p.requires_grad]
    param_groups = [{"params": main_params, "lr": float(base_lr)}]
    if len(aux_params) > 0:
        param_groups.append({"params": aux_params, "lr": float(base_lr) * float(artifact_aux_lr_scale)})
    if getattr(model, "artifact_domain_classifier", None) is not None:
        domain_params = [p for p in model.artifact_domain_classifier.parameters() if p.requires_grad]
        if len(domain_params) > 0:
            param_groups.append({"params": domain_params, "lr": float(base_lr) * float(artifact_domain_lr_scale)})
    return optim.AdamW(param_groups, weight_decay=0.02)


def _average_precision_from_arrays(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    if y_true.size == 0:
        return float("nan")
    pos_total = int((y_true == 1).sum())
    if pos_total == 0:
        return float("nan")
    if sk_average_precision_score is not None:
        try:
            return float(sk_average_precision_score(y_true, y_score))
        except Exception:
            pass
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    pos_mask = y_sorted == 1
    precision_at_k = np.cumsum(pos_mask) / (np.arange(len(y_sorted)) + 1.0)
    return float(precision_at_k[pos_mask].sum() / pos_total)


def _binary_accuracy(df: pd.DataFrame, label_value: int, pred_value: int) -> float:
    subset = df[df["label"] == label_value]
    return float((subset["pred"] == pred_value).mean()) if len(subset) > 0 else float("nan")


def _compute_threshold_metrics(df: pd.DataFrame, prob_col: str, threshold: float) -> Dict[str, float]:
    preds = (df[prob_col].values > threshold).astype(np.int64)
    label = df["label"].values.astype(np.int64)
    real_mask = label == 0
    fake_mask = label == 1
    accuracy = float((preds == label).mean()) if len(label) > 0 else float("nan")
    real_acc = float((preds[real_mask] == 0).mean()) if real_mask.any() else float("nan")
    fake_acc = float((preds[fake_mask] == 1).mean()) if fake_mask.any() else float("nan")
    balanced_acc = float(np.nanmean([real_acc, fake_acc]))
    tp = int(((preds == 1) & (label == 1)).sum())
    pred_pos = int((preds == 1).sum())
    precision = float(tp / max(pred_pos, 1))
    recall = fake_acc
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    ap = _average_precision_from_arrays(label, df[prob_col].values)
    return {
        "accuracy": accuracy,
        "ap": ap,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "balanced_acc": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _compute_group_macro_metrics(df: pd.DataFrame, prob_col: str, threshold: float, group_by: str = "model") -> Dict[str, float]:
    rows = []
    for _, g in df.groupby(group_by, dropna=False):
        rows.append(_compute_threshold_metrics(g, prob_col, threshold))
    if not rows:
        return {"macro_accuracy": float("nan"), "macro_balanced_acc": float("nan"), "macro_fake_acc": float("nan"), "macro_f1": float("nan")}
    metrics_df = pd.DataFrame(rows)
    return {
        "macro_accuracy": float(metrics_df["accuracy"].mean()),
        "macro_balanced_acc": float(metrics_df["balanced_acc"].mean()),
        "macro_fake_acc": float(metrics_df["fake_acc"].mean()),
        "macro_f1": float(metrics_df["f1"].mean()),
    }


def _compute_monitor_metric_from_raw_df(raw_df: pd.DataFrame, prob_col: str, threshold: float = 0.5) -> Dict[str, float]:
    overall = _compute_threshold_metrics(raw_df, prob_col, threshold)
    macro = _compute_group_macro_metrics(raw_df, prob_col, threshold, group_by="model")
    out = dict(overall)
    out.update(macro)
    return out


def _evaluate_one_head_from_df(df: pd.DataFrame, prob_col: str, threshold: float, group_by: str) -> pd.DataFrame:
    preds = (df[prob_col].values > threshold).astype(np.int64)
    tmp = df.copy()
    tmp["pred"] = preds
    rows = []
    for group_name, g in tmp.groupby(group_by):
        rows.append({
            group_by: group_name,
            "accuracy": float((g["label"] == g["pred"]).mean()),
            "ap": _average_precision_from_arrays(g["label"].values, g[prob_col].values),
            "num_samples": int(len(g)),
            "real_acc": _binary_accuracy(g, 0, 0),
            "fake_acc": _binary_accuracy(g, 1, 1),
        })
    out = pd.DataFrame(rows).sort_values("accuracy") if rows else pd.DataFrame(columns=[group_by, "accuracy", "ap", "num_samples", "real_acc", "fake_acc"])
    extra_rows = []
    if len(out) > 0:
        extra_rows.append({
            group_by: "Average",
            "accuracy": float(out["accuracy"].mean()),
            "ap": float(out["ap"].mean()),
            "num_samples": int(len(out)),
            "real_acc": float(out["real_acc"].mean()),
            "fake_acc": float(out["fake_acc"].mean()),
        })
    extra_rows.append({
        group_by: "Overall",
        "accuracy": float((tmp["label"] == tmp["pred"]).mean()) if len(tmp) > 0 else float("nan"),
        "ap": _average_precision_from_arrays(tmp["label"].values, tmp[prob_col].values) if len(tmp) > 0 else float("nan"),
        "num_samples": int(len(tmp)),
        "real_acc": _binary_accuracy(tmp, 0, 0),
        "fake_acc": _binary_accuracy(tmp, 1, 1),
    })
    return pd.concat([out, pd.DataFrame(extra_rows)], ignore_index=True)


def _split_group_report(df: pd.DataFrame, group_col: str = "model") -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    if df.empty:
        return df.copy(), {}
    mask = ~df[group_col].astype(str).isin(["Average", "Overall"])
    per_group_df = df.loc[mask].copy().reset_index(drop=True)
    summary = {}
    for _, row in per_group_df.iterrows():
        group_name = str(row[group_col])
        summary[group_name] = {
            "accuracy": float(row["accuracy"]),
            "ap": float(row["ap"]),
            "num_samples": int(row["num_samples"]),
            "real_acc": float(row["real_acc"]),
            "fake_acc": float(row["fake_acc"]),
        }
    return per_group_df, summary


def _print_per_model_report(title: str, df: pd.DataFrame, group_col: str = "model"):
    per_group_df, _ = _split_group_report(df, group_col=group_col)
    print(f"\n=== {title} ===")
    if per_group_df.empty:
        print("(empty)")
        return
    display_df = per_group_df.copy()
    for col in ["accuracy", "ap", "real_acc", "fake_acc"]:
        display_df[col] = display_df[col].map(lambda v: f"{float(v):.6f}")
    print(display_df.to_string(index=False))


def search_best_threshold(df: pd.DataFrame, prob_col: str, metric: str = "balanced_acc", threshold_min: float = 0.05, threshold_max: float = 0.95, num_steps: int = 181) -> Tuple[float, pd.DataFrame]:
    thresholds = np.linspace(threshold_min, threshold_max, num_steps)
    rows = []
    for t in thresholds:
        stats = _compute_threshold_metrics(df, prob_col, float(t))
        if metric.startswith("macro_"):
            stats.update(_compute_group_macro_metrics(df, prob_col, float(t), group_by="model"))
        rows.append({"threshold": float(t), **stats})
    curve_df = pd.DataFrame(rows)
    best_idx = curve_df[metric].idxmax()
    return float(curve_df.loc[best_idx, "threshold"]), curve_df


def move_inputs_to_device(inputs, device: str):
    return inputs.to(device, non_blocking=True)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: str, show_progress: bool = False, progress_desc: str = "predict") -> pd.DataFrame:
    probs, logits, labels, categories, models = [], [], [], [], []
    model.eval()
    progress = tqdm(loader, desc=progress_desc, leave=False, dynamic_ncols=True) if show_progress else loader
    for batch in progress:
        if batch is None:
            continue
        x, y, c = batch
        x = move_inputs_to_device(x, device)
        outputs = model(x, return_details=True)
        logit = outputs["artifact_output"].view(-1).detach().float().cpu().numpy()
        prob = torch.sigmoid(outputs["artifact_output"].view(-1)).detach().float().cpu().numpy()
        c_list = [str(v) for v in c]
        m_list = [str(v).replace("\\", "/").split("/")[0] for v in c_list]
        y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
        probs.append(prob)
        logits.append(logit)
        labels.append(y_np)
        categories.append(np.asarray(c_list, dtype=object))
        models.append(np.asarray(m_list, dtype=object))
    raw_df = pd.DataFrame({
        "category": np.concatenate(categories),
        "model": np.concatenate(models),
        "label": np.concatenate(labels).astype(np.int64),
        "prob": np.concatenate(probs),
        "logit": np.concatenate(logits),
    })
    raw_df["label_name"] = raw_df["label"].map({0: "real", 1: "fake"}).fillna("unknown")
    return raw_df


def maybe_save_test_reports(result_dir: str, split_name: str, raw_df: pd.DataFrame, curve_df: pd.DataFrame, report_by_model: pd.DataFrame):
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(os.path.join(result_dir, f"{split_name}_raw_predictions.csv"), index=False)
    curve_df.to_csv(os.path.join(result_dir, f"{split_name}_threshold_curve.csv"), index=False)
    report_by_model.to_csv(os.path.join(result_dir, f"{split_name}_by_model.csv"), index=False)


def evaluate_test_split(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    val_threshold: float,
    threshold_metric: str,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
    show_progress: bool = False,
    progress_desc: str = "test-predict",
) -> Tuple[Dict[str, object], Dict[str, pd.DataFrame]]:
    test_raw = collect_predictions(model, test_loader, device, show_progress=show_progress, progress_desc=progress_desc)
    test_best_thr, test_curve = search_best_threshold(
        test_raw,
        "prob",
        metric=str(threshold_metric),
        threshold_min=float(threshold_min),
        threshold_max=float(threshold_max),
        num_steps=int(threshold_steps),
    )
    test_fixed_by_model = _evaluate_one_head_from_df(test_raw, "prob", 0.5, "model")
    test_valselected_by_model = _evaluate_one_head_from_df(test_raw, "prob", float(val_threshold), "model")
    test_search_by_model = _evaluate_one_head_from_df(test_raw, "prob", float(test_best_thr), "model")
    _, test_fixed_per_model_summary = _split_group_report(test_fixed_by_model, group_col="model")
    _, test_valselected_per_model_summary = _split_group_report(test_valselected_by_model, group_col="model")
    _, test_search_per_model_summary = _split_group_report(test_search_by_model, group_col="model")
    metrics = {
        "test_threshold": float(test_best_thr),
        "test_search_metrics": _compute_monitor_metric_from_raw_df(test_raw, "prob", threshold=float(test_best_thr)),
        "test_valselected_metrics": _compute_monitor_metric_from_raw_df(test_raw, "prob", threshold=float(val_threshold)),
        "test_fixed05_metrics": _compute_monitor_metric_from_raw_df(test_raw, "prob", threshold=0.5),
        "per_model_summary": {
            "test_search": test_search_per_model_summary,
            "test_valselected": test_valselected_per_model_summary,
            "test_fixed05": test_fixed_per_model_summary,
        },
    }
    artifacts = {
        "test_raw": test_raw,
        "test_curve": test_curve,
        "test_fixed_by_model": test_fixed_by_model,
        "test_valselected_by_model": test_valselected_by_model,
        "test_search_by_model": test_search_by_model,
    }
    return metrics, artifacts


def _safe_state_dict(obj):
    if obj is None:
        return None
    try:
        return obj.state_dict()
    except Exception:
        return None


def build_checkpoint_payload(model: ArtifactOnlyBranchDetector, optimizer: Optional[optim.Optimizer], scheduler, epoch: int, best_val_acc: float, train_loss: Optional[float], train_acc: Optional[float], val_loss: Optional[float], val_acc: Optional[float], extra_meta: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    base_model = unwrap_model(model)
    payload = {
        "epoch": int(epoch),
        "best_val_acc": float(best_val_acc),
        "train_mode": "artifact_only",
        "monitor_head": "artifact",
        "train_loss": None if train_loss is None else float(train_loss),
        "train_acc": None if train_acc is None else float(train_acc),
        "val_loss": None if val_loss is None else float(val_loss),
        "val_acc": None if val_acc is None else float(val_acc),
        "optimizer_state_dict": _safe_state_dict(optimizer),
        "scheduler_state_dict": _safe_state_dict(scheduler),
        "checkpoint_type": "artifact",
        "artifact_extractor_state_dict": base_model.artifact_extractor.state_dict(),
        "artifact_classifier_state_dict": base_model.artifact_classifier.state_dict(),
        "artifact_aux_projector_state_dict": base_model.artifact_aux_projector.state_dict(),
        "artifact_branches": list(getattr(base_model, "artifact_branches", DEFAULT_ARTIFACT_BRANCHES)),
        "artifact_domain_num_domains": int(getattr(base_model, "artifact_domain_num_domains", 0) or 0),
        "artifact_domain_hidden_dim": int(getattr(base_model, "artifact_domain_hidden_dim", 0) or 0),
        "artifact_domain_classifier_state_dict": None if getattr(base_model, "artifact_domain_classifier", None) is None else base_model.artifact_domain_classifier.state_dict(),
    }
    if isinstance(extra_meta, dict) and extra_meta:
        payload["artifact_build_meta"] = dict(extra_meta)
    return payload


def load_checkpoint_for_resume(model: ArtifactOnlyBranchDetector, checkpoint_path: str, device: str, optimizer=None, scheduler=None, load_optimizer_state: bool = True, load_scheduler_state: bool = False) -> Dict[str, object]:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    base_model = unwrap_model(model)
    base_model.artifact_extractor.load_state_dict(ckpt["artifact_extractor_state_dict"], strict=False)
    base_model.artifact_classifier.load_state_dict(ckpt["artifact_classifier_state_dict"], strict=False)

    aux_projector_state = ckpt.get("artifact_aux_projector_state_dict", None)
    if isinstance(aux_projector_state, dict):
        base_model.artifact_aux_projector.load_state_dict(aux_projector_state, strict=False)

    # Important resume fix:
    # The old code always re-created artifact_domain_classifier after model.to(device).
    # That newly created LayerNorm/Linear module stayed on CPU and was also detached from
    # optimizer param groups, causing cuda/cpu mismatch during the next forward pass.
    # Here we reuse the existing domain classifier whenever dimensions match.
    domain_num = int(ckpt.get("artifact_domain_num_domains", 0) or 0)
    domain_hidden_dim = int(ckpt.get("artifact_domain_hidden_dim", 128) or 128)
    if domain_num > 1:
        existing_domain = getattr(base_model, "artifact_domain_classifier", None)
        existing_num = int(getattr(base_model, "artifact_domain_num_domains", 0) or 0)
        existing_hidden = int(getattr(base_model, "artifact_domain_hidden_dim", 0) or 0)
        if existing_domain is None or existing_num != domain_num or existing_hidden != domain_hidden_dim:
            base_model.configure_artifact_domain_classifier(domain_num, hidden_dim=domain_hidden_dim)
        if getattr(base_model, "artifact_domain_classifier", None) is not None:
            base_model.artifact_domain_classifier.to(device)
        domain_state = ckpt.get("artifact_domain_classifier_state_dict", None)
        if isinstance(domain_state, dict) and base_model.artifact_domain_classifier is not None:
            base_model.artifact_domain_classifier.load_state_dict(domain_state, strict=False)

    # Ensure every parameter/buffer is on the training device after checkpoint loading.
    base_model.to(device)

    if load_optimizer_state and optimizer is not None and isinstance(ckpt.get("optimizer_state_dict"), dict):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # Optimizer state tensors loaded from checkpoint must also be moved to the current device.
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            ckpt["optimizer_restored"] = True
        except Exception as exc:
            ckpt["optimizer_restored"] = False
            ckpt["optimizer_restore_error"] = str(exc)
    else:
        ckpt["optimizer_restored"] = False

    if load_scheduler_state and scheduler is not None and isinstance(ckpt.get("scheduler_state_dict"), dict):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            ckpt["scheduler_restored"] = True
        except Exception as exc:
            ckpt["scheduler_restored"] = False
            ckpt["scheduler_restore_error"] = str(exc)
    else:
        ckpt["scheduler_restored"] = False
    return ckpt


def maybe_wrap_dataparallel(model: nn.Module, device: str) -> nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        model = model.to(device)
    return model


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: ArtifactOnlyLoss,
    optimizer: optim.Optimizer,
    device: str,
    use_amp: bool,
    amp_dtype: str,
    grad_clip_norm: float,
    artifact_supcon_weight: float,
    artifact_supcon_temperature: float,
    artifact_domain_adv_weight: float,
    artifact_domain_adv_lambda: float,
    artifact_domain_to_idx: Optional[Dict[str, int]],
    show_progress: bool,
    epoch_desc: str,
    scaler=None,
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    model.train()
    use_scaler = scaler is not None

    run_loss = 0.0
    correct = 0
    total = 0
    supcon_meter = 0.0
    domain_meter = 0.0
    cat_stats = defaultdict(lambda: {"count": 0, "err": 0, "fake_count": 0, "fake_err": 0})
    pbar = tqdm(train_loader, desc=epoch_desc, leave=False, dynamic_ncols=True) if show_progress else train_loader
    for batch in pbar:
        if batch is None:
            continue
        inputs, labels, categories = batch
        inputs = move_inputs_to_device(inputs, device)
        labels = labels.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.startswith("cuda"):
            dtype = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(inputs, return_details=True)
                loss, _ = criterion(outputs, labels)
                aux_loss, aux_info = compute_artifact_aux_losses(
                    outputs=outputs,
                    labels=labels,
                    categories=categories,
                    model=model,
                    artifact_supcon_weight=artifact_supcon_weight,
                    artifact_supcon_temperature=artifact_supcon_temperature,
                    artifact_domain_adv_weight=artifact_domain_adv_weight,
                    artifact_domain_adv_lambda=artifact_domain_adv_lambda,
                    artifact_domain_to_idx=artifact_domain_to_idx,
                )
                loss = loss + aux_loss
        else:
            outputs = model(inputs, return_details=True)
            loss, _ = criterion(outputs, labels)
            aux_loss, aux_info = compute_artifact_aux_losses(
                outputs=outputs,
                labels=labels,
                categories=categories,
                model=model,
                artifact_supcon_weight=artifact_supcon_weight,
                artifact_supcon_temperature=artifact_supcon_temperature,
                artifact_domain_adv_weight=artifact_domain_adv_weight,
                artifact_domain_adv_lambda=artifact_domain_adv_lambda,
                artifact_domain_to_idx=artifact_domain_to_idx,
            )
            loss = loss + aux_loss

        if not torch.isfinite(loss):
            continue

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        run_loss += float(loss.item()) * labels.numel()
        supcon_meter += float(aux_info.get("supcon_loss", 0.0)) * labels.numel()
        domain_meter += float(aux_info.get("domain_loss", 0.0)) * labels.numel()
        logits = outputs["artifact_output"].detach().float().view(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        correct += int((preds == labels.long().view(-1)).sum().item())
        total += int(labels.numel())

        for cat, y, pred in zip(categories, labels.long().view(-1).cpu().tolist(), preds.cpu().tolist()):
            st = cat_stats[str(cat).lower()]
            st["count"] += 1
            st["err"] += int(y != pred)
            if int(y) == 1:
                st["fake_count"] += 1
                st["fake_err"] += int(y != pred)

        if show_progress and hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                loss=f"{run_loss / max(total, 1):.4f}",
                acc=f"{100.0 * correct / max(total, 1):.2f}%",
                supcon=f"{supcon_meter / max(total, 1):.4f}",
                domain=f"{domain_meter / max(total, 1):.4f}",
                sup_w=f"{artifact_supcon_weight:.3f}",
                dom_w=f"{artifact_domain_adv_weight:.3f}",
                artifact_logit=f"{outputs['artifact_output'].detach().float().view(-1).mean().item():.4f}"
            )

    epoch_err_stats = {}
    for cat, st in cat_stats.items():
        err_rate = st["err"] / max(st["count"], 1)
        fake_err_rate = st["fake_err"] / max(st["fake_count"], 1) if st["fake_count"] > 0 else err_rate
        epoch_err_stats[cat] = {"err_rate": err_rate, "fake_err_rate": fake_err_rate}

    return run_loss / max(total, 1), 100.0 * correct / max(total, 1), epoch_err_stats


@torch.no_grad()
def compute_val_loss_and_acc(model: nn.Module, val_loader: DataLoader, criterion: ArtifactOnlyLoss, device: str, use_amp: bool, amp_dtype: str, show_progress: bool, epoch_desc: str) -> Tuple[float, float]:
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    pbar = tqdm(val_loader, desc=epoch_desc, leave=False, dynamic_ncols=True) if show_progress else val_loader
    for batch in pbar:
        if batch is None:
            continue
        inputs, labels, _ = batch
        inputs = move_inputs_to_device(inputs, device)
        labels = labels.to(device, non_blocking=True).float()
        if use_amp and device.startswith("cuda"):
            dtype = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(inputs, return_details=True)
                loss, _ = criterion(outputs, labels)
        else:
            outputs = model(inputs, return_details=True)
            loss, _ = criterion(outputs, labels)
        val_loss_sum += float(loss.item()) * labels.numel()
        preds = (torch.sigmoid(outputs["artifact_output"].float().view(-1)) > 0.5).long()
        val_correct += int((preds == labels.long().view(-1)).sum().item())
        val_total += int(labels.numel())
    return val_loss_sum / max(val_total, 1), 100.0 * val_correct / max(val_total, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Useful3 artifact-only trainer: spectral_mag + wavelet + npr only")
    parser.add_argument("--train_root", type=str, default="./dataset/train")
    parser.add_argument("--val_root", type=str, default="./dataset/val")
    parser.add_argument("--test_root", type=str, default="./dataset/test")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/v13_artifact_only_minimal.pth")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_scheduler", action="store_true")
    parser.add_argument("--init_artifact_checkpoint", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=0, help="0 uses --batch_size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact_init_seed", type=int, default=-1)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--artifact_aux_lr_scale", type=float, default=0.5)
    parser.add_argument("--artifact_domain_lr_scale", type=float, default=0.25)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--checkpoint_metric", type=str, default="macro_accuracy", choices=["balanced_acc", "accuracy", "fake_acc", "f1", "macro_accuracy", "macro_balanced_acc", "macro_fake_acc", "macro_f1"])
    parser.add_argument("--threshold_metric", type=str, default="macro_accuracy", choices=["balanced_acc", "accuracy", "fake_acc", "f1", "macro_accuracy", "macro_balanced_acc", "macro_fake_acc", "macro_f1"])
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_steps", type=int, default=181)
    parser.add_argument("--hard_mining_dynamic_boost", type=float, default=2.0)
    parser.add_argument("--hard_mining_start_epoch", type=int, default=5)
    parser.add_argument("--hard_mining_ramp_epochs", type=int, default=4)
    parser.add_argument("--artifact_feature_dim", type=int, default=192)
    parser.add_argument("--artifact_token_dim", type=int, default=128)
    parser.add_argument("--token_pool_size", type=int, default=14)
    parser.add_argument("--artifact_feature_dropout", type=float, default=0.15)
    parser.add_argument("--artifact_aux_proj_hidden_dim", type=int, default=192)
    parser.add_argument("--artifact_aux_proj_dim", type=int, default=128)
    parser.add_argument("--artifact_aux_dropout", type=float, default=0.05)
    parser.add_argument("--ablation_preset", type=str, default="useful3", choices=["useful3", "all_old_no_recon", "all_old_with_recon"])
    parser.add_argument("--artifact_branches", nargs="+", default=None, choices=list(ALL_ARTIFACT_BRANCHES))
    parser.add_argument("--npr_scales", nargs="+", type=float, default=[0.25, 0.5, 0.75], help="MS-NPR downsample scales. Use '--npr_scales 0.5' for the original single-scale NPR baseline.")
    parser.set_defaults(raw_artifact_inputs=True)
    parser.add_argument("--raw_artifact_inputs", dest="raw_artifact_inputs", action="store_true", help="Use denormalized [0,1] RGB inputs for spectral/wavelet/laplacian artifact operators. This is the default.")
    parser.add_argument("--normalized_artifact_inputs", dest="raw_artifact_inputs", action="store_false", help="Use normalized CLIP-space inputs for spectral/wavelet/laplacian artifact operators to reproduce the old behavior.")
    parser.add_argument("--reconstruction_vae_path", type=str, default="")
    parser.add_argument("--reconstruction_vae_subfolder", type=str, default="")
    parser.add_argument("--reconstruction_vae_dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--reconstruction_use_fft_residual", action="store_true")
    parser.add_argument("--artifact_supcon_weight", type=float, default=0.05)
    parser.add_argument("--artifact_supcon_temperature", type=float, default=0.07)
    parser.add_argument("--artifact_supcon_start_epoch", type=int, default=2)
    parser.add_argument("--artifact_supcon_ramp_epochs", type=int, default=2)
    parser.add_argument("--artifact_domain_adv_weight", type=float, default=0.05)
    parser.add_argument("--artifact_domain_adv_lambda", type=float, default=1.0)
    parser.add_argument("--artifact_domain_adv_start_epoch", type=int, default=2)
    parser.add_argument("--artifact_domain_adv_ramp_epochs", type=int, default=2)
    parser.add_argument("--artifact_domain_hidden_dim", type=int, default=128)
    parser.add_argument("--artifact_pos_scale", type=float, default=1.5)
    parser.add_argument("--verify_images", action="store_true")
    parser.add_argument("--save_test_reports", action="store_true")
    parser.add_argument("--periodic_eval_interval", type=int, default=5, help="Save current checkpoint and run test every N epochs; <=0 disables")
    parser.add_argument("--periodic_eval_save_reports", action="store_true", help="Save raw predictions and threshold curves for periodic evaluations")
    parser.add_argument("--show_progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.npr_scales = list(normalize_npr_scales(args.npr_scales))
    setup_seed(int(args.seed))
    device = resolve_device(args.device)
    persistent = int(args.num_workers) > 0 and os.name != "nt"

    branches, reconstruction_use_fft_residual = resolve_artifact_branches_for_run(args)
    print(f"[Useful3+B1 MS-NPR] active artifact branches: {branches}")
    print(f"[Useful3+B1 MS-NPR] npr_scales: {args.npr_scales}")

    image_size = validate_image_size(args.image_size)
    train_ds = None
    if not args.test_only:
        train_ds = ImageForgeryDataset(args.train_root, transform=get_artifact_train_transform(image_size), verify_images=args.verify_images)
    val_ds = ImageForgeryDataset(args.val_root, transform=get_val_transform(image_size), verify_images=args.verify_images)
    test_ds = ImageForgeryDataset(args.test_root, transform=get_val_transform(image_size), verify_images=args.verify_images) if args.test_root and os.path.exists(args.test_root) else None

    artifact_domain_to_idx = None
    if not args.test_only and float(args.artifact_domain_adv_weight) > 0.0:
        artifact_domain_to_idx = get_artifact_domain_to_idx(train_ds)

    model = build_pure_artifact_model(
        args=args,
        branches=branches,
        reconstruction_use_fft_residual=reconstruction_use_fft_residual,
        artifact_domain_to_idx=artifact_domain_to_idx,
    ).to(device)

    freeze_for_artifact_only(model)
    print(
        "Artifact-only config | "
        f"artifact_init_seed={resolve_artifact_init_seed(args.seed, args.artifact_init_seed)}, "
        f"ablation_preset={str(args.ablation_preset)}, "
        f"branches={branches}, "
        f"npr_scales={args.npr_scales}, "
        f"raw_artifact_inputs={bool(args.raw_artifact_inputs)}, "
        f"reconstruction_use_fft_residual={bool(reconstruction_use_fft_residual)}, "
        f"aux_proj={int(args.artifact_aux_proj_hidden_dim)}->{int(args.artifact_aux_proj_dim)}, "
        f"supcon={float(args.artifact_supcon_weight)}@start{int(args.artifact_supcon_start_epoch)}/ramp{int(args.artifact_supcon_ramp_epochs)}, "
        f"domain_adv={float(args.artifact_domain_adv_weight)}@start{int(args.artifact_domain_adv_start_epoch)}/ramp{int(args.artifact_domain_adv_ramp_epochs)}, "
        f"hard_mining={float(args.hard_mining_dynamic_boost)}@start{int(args.hard_mining_start_epoch)}/ramp{int(args.hard_mining_ramp_epochs)}, "
        f"image_size={int(args.image_size)}, "
        f"pos_scale={float(args.artifact_pos_scale)}, "
        f"periodic_eval_interval={int(args.periodic_eval_interval)}, "
        f"scheduler={str(args.scheduler)}"
    )
    optimizer = None
    scheduler = None
    if not args.test_only:
        optimizer = build_optimizer(
            model,
            base_lr=float(args.base_lr),
            artifact_aux_lr_scale=float(args.artifact_aux_lr_scale),
            artifact_domain_lr_scale=float(args.artifact_domain_lr_scale),
        )
        if str(args.scheduler).lower() == "cosine":
            total_epochs = max(int(args.epochs), 1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=float(args.min_lr))
        else:
            scheduler = None

    if args.init_artifact_checkpoint and not args.resume:
        load_checkpoint_for_resume(model, args.init_artifact_checkpoint, device, optimizer=None, scheduler=None, load_optimizer_state=False, load_scheduler_state=False)

    start_epoch = 0
    best_monitor_value = float("-inf")
    if args.resume:
        ckpt = load_checkpoint_for_resume(model, args.checkpoint, device, optimizer=optimizer, scheduler=scheduler if args.resume_scheduler else None, load_optimizer_state=True, load_scheduler_state=bool(args.resume_scheduler))
        start_epoch = int(ckpt.get("epoch", 0) or 0)
        artifact_build_meta = ckpt.get("artifact_build_meta", {}) if isinstance(ckpt.get("artifact_build_meta", {}), dict) else {}
        best_monitor_value = float(
            artifact_build_meta.get("best_checkpoint_metric_value",
            ckpt.get("best_val_acc", float("-inf"))) or float("-inf")
        )

    model = maybe_wrap_dataparallel(model, device)
    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    result_dir = str(Path(args.checkpoint).with_suffix(""))

    if not args.test_only:
        sampler, _, group_counts, label_counts = build_stage12_sampler(train_ds, balance_labels=True)
        print("✅ Stage12 sampler built")
        print(f"  group_counts(top10): {sorted(group_counts.items(), key=lambda x: -x[1])[:10]}")
        print(f"  label_counts: {label_counts}")
        train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), sampler=sampler, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(), persistent_workers=persistent, drop_last=True, collate_fn=forgiving_collate)
        val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(), persistent_workers=persistent, collate_fn=forgiving_collate)
        criterion = ArtifactOnlyLoss(class_weights=compute_class_weights_from_labels(train_ds.labels).to(device), pos_scale=args.artifact_pos_scale, focal_gamma=2.0, focal_alpha=0.65)

        patience_counter = 0
        use_scaler = bool(args.use_amp) and device.startswith("cuda") and str(args.amp_dtype).lower() in {"fp16", "float16"}
        scaler = create_grad_scaler(use_scaler)

        for epoch in range(start_epoch, int(args.epochs)):
            active_supcon_weight = compute_delayed_weight(
                epoch_idx=epoch,
                target_weight=float(args.artifact_supcon_weight),
                start_epoch=int(args.artifact_supcon_start_epoch),
                ramp_epochs=int(args.artifact_supcon_ramp_epochs),
            )
            active_domain_adv_weight = compute_delayed_weight(
                epoch_idx=epoch,
                target_weight=float(args.artifact_domain_adv_weight),
                start_epoch=int(args.artifact_domain_adv_start_epoch),
                ramp_epochs=int(args.artifact_domain_adv_ramp_epochs),
            )
            active_hard_mining_boost = compute_delayed_weight(
                epoch_idx=epoch,
                target_weight=float(args.hard_mining_dynamic_boost),
                start_epoch=int(args.hard_mining_start_epoch),
                ramp_epochs=int(args.hard_mining_ramp_epochs),
            )

            train_loss, train_acc, epoch_err_stats = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                use_amp=bool(args.use_amp),
                amp_dtype=str(args.amp_dtype),
                grad_clip_norm=float(args.grad_clip),
                artifact_supcon_weight=active_supcon_weight,
                artifact_supcon_temperature=float(args.artifact_supcon_temperature),
                artifact_domain_adv_weight=active_domain_adv_weight,
                artifact_domain_adv_lambda=float(args.artifact_domain_adv_lambda),
                artifact_domain_to_idx=artifact_domain_to_idx,
                show_progress=bool(args.show_progress),
                epoch_desc=f"Epoch {epoch+1}/{int(args.epochs)} - Train",
                scaler=scaler,
            )
            val_loss, val_acc = compute_val_loss_and_acc(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=bool(args.use_amp),
                amp_dtype=str(args.amp_dtype),
                show_progress=bool(args.show_progress),
                epoch_desc=f"Epoch {epoch+1}/{int(args.epochs)} - Val",
            )
            val_raw_epoch = collect_predictions(
                model,
                val_loader,
                device,
                show_progress=bool(args.show_progress),
                progress_desc=f"Epoch {epoch+1}/{int(args.epochs)} - ValPredict",
            )
            val_epoch_thr, val_curve_epoch = search_best_threshold(
                val_raw_epoch,
                "prob",
                metric=str(args.threshold_metric),
                threshold_min=float(args.threshold_min),
                threshold_max=float(args.threshold_max),
                num_steps=int(args.threshold_steps),
            )
            val_epoch_metrics = _compute_monitor_metric_from_raw_df(
                val_raw_epoch,
                "prob",
                threshold=float(val_epoch_thr),
            )
            monitor_value = float(val_epoch_metrics[str(args.checkpoint_metric)])

            if scheduler is not None:
                try:
                    scheduler.step()
                except TypeError:
                    scheduler.step(val_acc)
            print(
                f"Epoch {epoch+1}/{int(args.epochs)} | "
                f"sup_w {active_supcon_weight:.4f} dom_w {active_domain_adv_weight:.4f} hm_w {active_hard_mining_boost:.4f} | "
                f"TrainLoss {train_loss:.5f} artifactAcc {train_acc:.2f}% | "
                f"ValLoss {val_loss:.5f} artifactAcc {val_acc:.2f}% | "
                f"ValThr {val_epoch_thr:.4f} {str(args.checkpoint_metric)} {monitor_value:.6f}"
            )

            should_stop = False
            if monitor_value > best_monitor_value:
                best_monitor_value = monitor_value
                patience_counter = 0
                torch.save(
                    build_checkpoint_payload(
                        unwrap_model(model),
                        optimizer,
                        scheduler,
                        epoch + 1,
                        best_monitor_value,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        extra_meta={
                            "artifact_init_seed": resolve_artifact_init_seed(args.seed, args.artifact_init_seed),
                            "ablation_preset": str(args.ablation_preset),
                            "image_size": int(args.image_size),
                            "npr_scales": [float(v) for v in args.npr_scales],
                            "artifact_pos_scale": float(args.artifact_pos_scale),
                            "artifact_supcon_weight": float(args.artifact_supcon_weight),
                            "artifact_supcon_start_epoch": int(args.artifact_supcon_start_epoch),
                            "artifact_supcon_ramp_epochs": int(args.artifact_supcon_ramp_epochs),
                            "artifact_domain_adv_weight": float(args.artifact_domain_adv_weight),
                            "artifact_domain_adv_start_epoch": int(args.artifact_domain_adv_start_epoch),
                            "artifact_domain_adv_ramp_epochs": int(args.artifact_domain_adv_ramp_epochs),
                            "artifact_aux_proj_hidden_dim": int(args.artifact_aux_proj_hidden_dim),
                            "artifact_aux_proj_dim": int(args.artifact_aux_proj_dim),
                            "artifact_aux_lr_scale": float(args.artifact_aux_lr_scale),
                            "artifact_domain_lr_scale": float(args.artifact_domain_lr_scale),
                            "hard_mining_dynamic_boost": float(args.hard_mining_dynamic_boost),
                            "hard_mining_start_epoch": int(args.hard_mining_start_epoch),
                            "hard_mining_ramp_epochs": int(args.hard_mining_ramp_epochs),
                            "scheduler": str(args.scheduler),
                            "min_lr": float(args.min_lr),
                            "reconstruction_use_fft_residual": bool(reconstruction_use_fft_residual),
                            "checkpoint_metric": str(args.checkpoint_metric),
                            "threshold_metric": str(args.threshold_metric),
                            "best_checkpoint_metric_value": float(best_monitor_value),
                            "best_checkpoint_threshold": float(val_epoch_thr),
                            "best_checkpoint_metrics": dict(val_epoch_metrics),
                        },
                    ),
                    args.checkpoint,
                )
                print(
                    f"  ✓ Saved best checkpoint (artifact): {args.checkpoint} "
                    f"(Best {str(args.checkpoint_metric)}: {best_monitor_value:.6f} @ thr={val_epoch_thr:.4f})"
                )
            else:
                patience_counter += 1
                if patience_counter >= int(args.patience):
                    print(
                        f"  ⏹ Early stopping requested at epoch {epoch+1} "
                        f"(best {str(args.checkpoint_metric)}: {best_monitor_value:.6f})"
                    )
                    should_stop = True

            periodic_interval = int(args.periodic_eval_interval)
            if periodic_interval > 0 and ((epoch + 1) % periodic_interval == 0):
                epoch_tag = f"epoch_{epoch + 1:03d}"
                epoch_dir = os.path.join(result_dir, epoch_tag)
                Path(epoch_dir).mkdir(parents=True, exist_ok=True)
                periodic_ckpt = os.path.join(epoch_dir, f"checkpoint_{epoch_tag}.pth")
                torch.save(
                    build_checkpoint_payload(
                        unwrap_model(model),
                        optimizer,
                        scheduler,
                        epoch + 1,
                        best_monitor_value,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        extra_meta={
                            "artifact_init_seed": resolve_artifact_init_seed(args.seed, args.artifact_init_seed),
                            "ablation_preset": str(args.ablation_preset),
                            "image_size": int(args.image_size),
                            "npr_scales": [float(v) for v in args.npr_scales],
                            "artifact_pos_scale": float(args.artifact_pos_scale),
                            "artifact_supcon_weight": float(args.artifact_supcon_weight),
                            "artifact_domain_adv_weight": float(args.artifact_domain_adv_weight),
                            "active_supcon_weight": float(active_supcon_weight),
                            "active_domain_adv_weight": float(active_domain_adv_weight),
                            "active_hard_mining_boost": float(active_hard_mining_boost),
                            "checkpoint_metric": str(args.checkpoint_metric),
                            "threshold_metric": str(args.threshold_metric),
                            "current_val_threshold": float(val_epoch_thr),
                            "current_val_metrics": dict(val_epoch_metrics),
                            "best_checkpoint_metric_value": float(best_monitor_value),
                        },
                    ),
                    periodic_ckpt,
                )
                periodic_summary = {
                    "epoch": int(epoch + 1),
                    "checkpoint": periodic_ckpt,
                    "image_size": int(args.image_size),
                    "artifact_pos_scale": float(args.artifact_pos_scale),
                    "artifact_supcon_weight": float(args.artifact_supcon_weight),
                    "artifact_domain_adv_weight": float(args.artifact_domain_adv_weight),
                    "active_supcon_weight": float(active_supcon_weight),
                    "active_domain_adv_weight": float(active_domain_adv_weight),
                    "active_hard_mining_boost": float(active_hard_mining_boost),
                    "val_threshold": float(val_epoch_thr),
                    "val_search_metrics": dict(val_epoch_metrics),
                    "val_fixed05_metrics": _compute_monitor_metric_from_raw_df(val_raw_epoch, "prob", threshold=0.5),
                }
                if test_ds is not None:
                    eval_batch_size_periodic = int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size)
                    periodic_test_loader = DataLoader(
                        test_ds,
                        batch_size=eval_batch_size_periodic,
                        shuffle=False,
                        num_workers=int(args.num_workers),
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=persistent,
                        collate_fn=forgiving_collate,
                    )
                    test_metrics_epoch, test_artifacts_epoch = evaluate_test_split(
                        model=model,
                        test_loader=periodic_test_loader,
                        device=device,
                        val_threshold=float(val_epoch_thr),
                        threshold_metric=str(args.threshold_metric),
                        threshold_min=float(args.threshold_min),
                        threshold_max=float(args.threshold_max),
                        threshold_steps=int(args.threshold_steps),
                        show_progress=bool(args.show_progress),
                        progress_desc=f"{epoch_tag}-test-predict",
                    )
                    periodic_summary.update(test_metrics_epoch)
                    test_valselected = test_metrics_epoch["test_valselected_metrics"]
                    test_search = test_metrics_epoch["test_search_metrics"]
                    print(
                        f"  ✓ Periodic save/test {epoch_tag}: {periodic_ckpt} | "
                        f"val_thr={float(val_epoch_thr):.4f} "
                        f"test_valsel_macro_bal={float(test_valselected['macro_balanced_acc']):.6f} "
                        f"test_search_macro_bal={float(test_search['macro_balanced_acc']):.6f}"
                    )
                    test_artifacts_epoch["test_valselected_by_model"].to_csv(os.path.join(epoch_dir, "test_valselected_by_model.csv"), index=False)
                    test_artifacts_epoch["test_search_by_model"].to_csv(os.path.join(epoch_dir, "test_search_by_model.csv"), index=False)
                    test_artifacts_epoch["test_fixed_by_model"].to_csv(os.path.join(epoch_dir, "test_fixed05_by_model.csv"), index=False)
                    if args.periodic_eval_save_reports or args.save_test_reports:
                        maybe_save_test_reports(epoch_dir, "test_fixed05", test_artifacts_epoch["test_raw"], test_artifacts_epoch["test_curve"], test_artifacts_epoch["test_fixed_by_model"])
                        maybe_save_test_reports(epoch_dir, "test_valselected", test_artifacts_epoch["test_raw"], test_artifacts_epoch["test_curve"], test_artifacts_epoch["test_valselected_by_model"])
                        maybe_save_test_reports(epoch_dir, "test_search", test_artifacts_epoch["test_raw"], test_artifacts_epoch["test_curve"], test_artifacts_epoch["test_search_by_model"])
                else:
                    print(f"  ✓ Periodic save {epoch_tag}: {periodic_ckpt} | test_root not found, skipped test")
                val_raw_epoch.to_csv(os.path.join(epoch_dir, "val_raw_predictions.csv"), index=False)
                val_curve_epoch.to_csv(os.path.join(epoch_dir, "val_threshold_curve.csv"), index=False)
                with open(os.path.join(epoch_dir, "periodic_eval_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(periodic_summary, f, ensure_ascii=False, indent=2)

            if isinstance(train_loader.sampler, WeightedRandomSampler) and active_hard_mining_boost > 0.0:
                update_sampler_for_hard_mining(train_loader.dataset, train_loader.sampler, epoch_err_stats, dynamic_boost=float(active_hard_mining_boost))
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if should_stop:
                break

    if os.path.exists(args.checkpoint):
        load_checkpoint_for_resume(model, args.checkpoint, device, optimizer=None, scheduler=None, load_optimizer_state=False, load_scheduler_state=False)

    eval_batch_size = int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size)
    val_loader_eval = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(), persistent_workers=persistent, collate_fn=forgiving_collate)
    val_raw = collect_predictions(model, val_loader_eval, device, show_progress=bool(args.show_progress), progress_desc="val-predict")
    val_best_thr, val_curve = search_best_threshold(val_raw, "prob", metric=str(args.threshold_metric), threshold_min=float(args.threshold_min), threshold_max=float(args.threshold_max), num_steps=int(args.threshold_steps))

    summary = {
        "checkpoint": str(args.checkpoint),
        "ablation_preset": str(args.ablation_preset),
        "artifact_branches": list(branches),
        "selected_threshold": float(val_best_thr),
        "checkpoint_metric": str(args.checkpoint_metric),
        "threshold_metric": str(args.threshold_metric),
        "image_size": int(args.image_size),
        "eval_batch_size": int(eval_batch_size),
        "periodic_eval_interval": int(args.periodic_eval_interval),
        "artifact_pos_scale": float(args.artifact_pos_scale),
        "train_sampler": "label_balanced_hard_mining",
        "hard_mining_dynamic_boost": float(args.hard_mining_dynamic_boost),
        "hard_mining_start_epoch": int(args.hard_mining_start_epoch),
        "hard_mining_ramp_epochs": int(args.hard_mining_ramp_epochs),
        "scheduler": str(args.scheduler),
        "min_lr": float(args.min_lr),
        "artifact_aux_proj_hidden_dim": int(args.artifact_aux_proj_hidden_dim),
        "artifact_aux_proj_dim": int(args.artifact_aux_proj_dim),
        "artifact_aux_lr_scale": float(args.artifact_aux_lr_scale),
        "artifact_domain_lr_scale": float(args.artifact_domain_lr_scale),
        "train_aug": "artifact_light_aug",
        "artifact_supcon_weight": float(args.artifact_supcon_weight),
        "artifact_supcon_temperature": float(args.artifact_supcon_temperature),
        "artifact_domain_adv_weight": float(args.artifact_domain_adv_weight),
        "artifact_domain_adv_lambda": float(args.artifact_domain_adv_lambda),
        "val_threshold": float(val_best_thr),
        "val_search_metrics": _compute_monitor_metric_from_raw_df(val_raw, "prob", threshold=float(val_best_thr)),
        "val_fixed05_metrics": _compute_monitor_metric_from_raw_df(val_raw, "prob", threshold=0.5),
        "val_checkpoint_selected_metric": float(_compute_monitor_metric_from_raw_df(val_raw, "prob", threshold=float(val_best_thr)).get(str(args.checkpoint_metric), float("nan"))),
    }

    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(), persistent_workers=persistent, collate_fn=forgiving_collate)
        test_metrics, test_artifacts = evaluate_test_split(
            model=model,
            test_loader=test_loader,
            device=device,
            val_threshold=float(val_best_thr),
            threshold_metric=str(args.threshold_metric),
            threshold_min=float(args.threshold_min),
            threshold_max=float(args.threshold_max),
            threshold_steps=int(args.threshold_steps),
            show_progress=bool(args.show_progress),
            progress_desc="test-predict",
        )
        summary.update(test_metrics)
        test_raw = test_artifacts["test_raw"]
        test_curve = test_artifacts["test_curve"]
        test_fixed_by_model = test_artifacts["test_fixed_by_model"]
        test_valselected_by_model = test_artifacts["test_valselected_by_model"]
        test_search_by_model = test_artifacts["test_search_by_model"]
        _print_per_model_report(f"test_search_by_model @ threshold={float(test_metrics['test_threshold']):.6f}", test_search_by_model, group_col="model")
        _print_per_model_report(f"test_valselected_by_model @ threshold={float(val_best_thr):.6f}", test_valselected_by_model, group_col="model")
        _print_per_model_report("test_fixed05_by_model @ threshold=0.500000", test_fixed_by_model, group_col="model")
        if args.save_test_reports:
            maybe_save_test_reports(result_dir, "test_fixed05", test_raw, test_curve, test_fixed_by_model)
            maybe_save_test_reports(result_dir, "test_valselected", test_raw, test_curve, test_valselected_by_model)
            maybe_save_test_reports(result_dir, "test_search", test_raw, test_curve, test_search_by_model)

    Path(result_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(result_dir, "artifact_only_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
