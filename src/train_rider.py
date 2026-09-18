#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBU = True
except Exception:
    HAS_ALBU = False

try:
    from sklearn.metrics import average_precision_score as sk_average_precision_score
except Exception:
    sk_average_precision_score = None

import clip

try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
FULL_MODEL_SEMANTIC_DROPOUT_DEFAULT = 0.10
FULL_MODEL_SEMANTIC_TOKEN_MASK_PROB_DEFAULT = 0.05
SEMANTIC_ONLY_SEMANTIC_DROPOUT_DEFAULT = 0.0
SEMANTIC_ONLY_SEMANTIC_TOKEN_MASK_PROB_DEFAULT = 0.0
DEFAULT_PATCH_SHUFFLE_PROB = 0.50
DEFAULT_NPR_SCALES = (0.25, 0.5, 0.75)

# Useful3+B1 MS-NPR artifact branch set from artifact_only_useful3_msnpr_from_v13.py.
DEFAULT_ARTIFACT_BRANCHES = ("spectral_mag", "wavelet", "npr")
ALL_ARTIFACT_BRANCHES = (
    "spectral_mag",
    "wavelet",
    "npr",
)


def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



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



def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if device_arg in {"auto", "cuda"}:
        return "cuda:0"
    return device_arg



def _normalize_path_key(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def _candidate_path_keys(path: str) -> List[str]:
    raw = str(path)
    norm = _normalize_path_key(raw)
    keys = [norm]
    try:
        keys.append(_normalize_path_key(os.path.abspath(raw)))
    except Exception:
        pass
    keys.append(os.path.basename(norm))
    out: List[str] = []
    seen = set()
    for key in keys:
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        v = float(value)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _logit_from_prob(prob: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return float(math.log(p / (1.0 - p)))


def load_clip_nn_score_csv(csv_path: str, score_column: str = "auto") -> Dict[str, float]:
    if not csv_path:
        return {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CLIP-NN CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CLIP-NN CSV: {csv_path}")
        fields = [str(c) for c in reader.fieldnames]
        lower_to_field = {c.lower(): c for c in fields}
        path_field = None
        for name in ["path", "image_path", "img_path", "filepath", "file", "image"]:
            if name in lower_to_field:
                path_field = lower_to_field[name]
                break
        if path_field is None:
            raise ValueError(f"CLIP-NN CSV must contain a path column. Found: {fields}")
        if score_column != "auto":
            if score_column not in fields:
                raise ValueError(f"Requested --clip_nn_score_column={score_column} not found in {csv_path}. Found: {fields}")
            score_field = score_column
            mode = "prob" if "prob" in score_column.lower() else "score"
        else:
            score_field = None
            mode = "score"
            for name in ["clip_nn_logit", "nn_logit", "logit"]:
                if name in lower_to_field:
                    score_field = lower_to_field[name]
                    mode = "score"
                    break
            if score_field is None:
                for name in ["clip_nn_score", "nn_score", "score"]:
                    if name in lower_to_field:
                        score_field = lower_to_field[name]
                        mode = "score"
                        break
            if score_field is None:
                for name in ["clip_nn_prob", "nn_prob", "prob", "probability"]:
                    if name in lower_to_field:
                        score_field = lower_to_field[name]
                        mode = "prob"
                        break
        if score_field is None:
            raise ValueError(f"CLIP-NN CSV must contain a score/logit/prob column. Found: {fields}")
        out: Dict[str, float] = {}
        basename_seen = defaultdict(int)
        rows = list(reader)
        for row in rows:
            path = str(row.get(path_field, ""))
            if not path:
                continue
            val = _safe_float(row.get(score_field), default=0.0)
            if mode == "prob":
                val = _logit_from_prob(val)
            norm = _normalize_path_key(path)
            out[norm] = val
            try:
                out[_normalize_path_key(os.path.abspath(path))] = val
            except Exception:
                pass
            basename_seen[os.path.basename(norm)] += 1
        for row in rows:
            path = str(row.get(path_field, ""))
            if not path:
                continue
            base = os.path.basename(_normalize_path_key(path))
            if basename_seen.get(base, 0) == 1:
                val = _safe_float(row.get(score_field), default=0.0)
                if mode == "prob":
                    val = _logit_from_prob(val)
                out[base] = val
        return out


class ClipNNScoreDataset(Dataset):
    def __init__(self, base_dataset: Dataset, score_map: Dict[str, float], missing_policy: str = "error", default_score: float = 0.0):
        self.base_dataset = base_dataset
        self.score_map = score_map or {}
        self.missing_policy = str(missing_policy).lower().strip()
        if self.missing_policy not in {"error", "zero"}:
            raise ValueError("missing_policy must be 'error' or 'zero'")
        self.default_score = float(default_score)
        for name in ["samples", "images", "labels", "categories", "models", "subsets", "root_dir"]:
            if hasattr(base_dataset, name):
                setattr(self, name, getattr(base_dataset, name))

    def __len__(self):
        return len(self.base_dataset)

    def _lookup_score(self, idx: int) -> float:
        path = None
        if hasattr(self.base_dataset, "samples"):
            try:
                path = str(self.base_dataset.samples[idx].get("path", ""))
            except Exception:
                path = None
        if not path and hasattr(self.base_dataset, "images"):
            try:
                path = str(self.base_dataset.images[idx])
            except Exception:
                path = None
        if path:
            for key in _candidate_path_keys(path):
                if key in self.score_map:
                    return float(self.score_map[key])
        if self.missing_policy == "zero":
            return float(self.default_score)
        raise KeyError(f"Missing CLIP-NN score for dataset index={idx}, path={path}")

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        if item is None:
            return None
        score = torch.tensor(float(self._lookup_score(idx)), dtype=torch.float32)
        if isinstance(item, tuple):
            return (*item, score)
        raise RuntimeError("Wrapped dataset must return a tuple")


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
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found under: {root_dir}")

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
        parts = [p for p in str(category).replace("\\", "/").strip("/").split("/") if p]
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


class SharedGeometryDualViewImageForgeryDataset(ImageForgeryDataset):
    def __init__(
        self,
        root_dir: str,
        shared_geometry_transform=None,
        semantic_post_transform=None,
        artifact_post_transform=None,
        verify_images: bool = False,
    ):
        super().__init__(root_dir=root_dir, transform=None, verify_images=verify_images)
        self.shared_geometry_transform = shared_geometry_transform
        self.semantic_post_transform = semantic_post_transform
        self.artifact_post_transform = artifact_post_transform if artifact_post_transform is not None else semantic_post_transform

    @staticmethod
    def _apply_transform(image: np.ndarray, transform):
        if transform is None:
            return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        out = transform(image=image)
        return out["image"] if isinstance(out, dict) and "image" in out else out

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = str(sample["path"])
        label = int(sample["label"])
        category = str(sample["category"])
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            if self.shared_geometry_transform is not None:
                shared = self.shared_geometry_transform(image=image)
                image = shared["image"] if isinstance(shared, dict) and "image" in shared else shared
            semantic_image = self._apply_transform(image, self.semantic_post_transform)
            artifact_image = self._apply_transform(image, self.artifact_post_transform)
            return {"semantic": semantic_image, "artifact": artifact_image}, label, category
        except Exception:
            return None



def forgiving_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)



def get_val_transform():
    if HAS_ALBU:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=list(CLIP_IMAGE_MEAN), std=list(CLIP_IMAGE_STD)),
            ToTensorV2(),
        ])

    class _Simple:
        def __call__(self, image):
            img = Image.fromarray(image).resize((224, 224))
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = (arr - np.array(CLIP_IMAGE_MEAN, dtype=np.float32)) / np.array(CLIP_IMAGE_STD, dtype=np.float32)
            return {"image": torch.from_numpy(arr).permute(2, 0, 1)}

    return _Simple()



def get_shared_geometry_transform():
    if HAS_ALBU:
        return A.ReplayCompose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.25),
            A.OneOf([
                A.Affine(scale=(0.85, 1.15), translate_percent=(0.08, 0.08), rotate=(-20, 20), shear=(-8, 8), p=1.0),
                A.Perspective(scale=(0.03, 0.08), p=1.0),
            ], p=0.25),
        ])

    class _Simple:
        def __call__(self, image):
            img = Image.fromarray(image).resize((224, 224))
            return {"image": np.asarray(img)}

    return _Simple()



def get_semantic_post_transform():
    if HAS_ALBU:
        return A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.NoOp(p=1.0),
            ], p=0.2),
            A.OneOf([
                A.ImageCompression(quality_range=(65, 100), p=1.0),
                A.NoOp(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.06), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.4), p=1.0),
                A.NoOp(p=1.0),
            ], p=0.25),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=1.0),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=1.0),
                A.NoOp(p=1.0),
            ], p=0.2),
            A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(8, 16), hole_width_range=(8, 16), fill=0, p=0.15),
            A.Normalize(mean=list(CLIP_IMAGE_MEAN), std=list(CLIP_IMAGE_STD)),
            ToTensorV2(),
        ])

    return get_val_transform()



def get_artifact_post_transform():
    if HAS_ALBU:
        return A.Compose([
            A.ImageCompression(quality_range=(85, 100), p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.15),
            A.Normalize(mean=list(CLIP_IMAGE_MEAN), std=list(CLIP_IMAGE_STD)),
            ToTensorV2(),
        ])

    return get_val_transform()



def build_train_dataset(root_dir: str, verify_images: bool = False):
    return SharedGeometryDualViewImageForgeryDataset(
        root_dir=root_dir,
        shared_geometry_transform=get_shared_geometry_transform(),
        semantic_post_transform=get_semantic_post_transform(),
        artifact_post_transform=get_artifact_post_transform(),
        verify_images=verify_images,
    )



def build_eval_dataset(root_dir: str, verify_images: bool = False):
    return ImageForgeryDataset(root_dir=root_dir, transform=get_val_transform(), verify_images=verify_images)



def load_clip_model(model_name: str, device: str):
    local_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")
    os.makedirs(local_model_dir, exist_ok=True)
    model, _ = clip.load(model_name, device=device, download_root=local_model_dir)
    return model



def _get_clip_visual_resblocks(clip_model) -> List[nn.Module]:
    visual = getattr(clip_model, "visual", None)
    transformer = getattr(visual, "transformer", None) if visual is not None else None
    resblocks = getattr(transformer, "resblocks", None) if transformer is not None else None
    return list(resblocks) if resblocks is not None else []



def cast_clip_to_fp32_for_finetune(clip_model: nn.Module) -> nn.Module:
    """Keep trainable CLIP weights in fp32 when CLIP is partially finetuned."""
    return clip_model.float()


def set_clip_finetune(clip_model, unfreeze_last_n: int = 0, train_layernorm: bool = True):
    for p in clip_model.parameters():
        p.requires_grad = False
    if unfreeze_last_n > 0:
        for blk in _get_clip_visual_resblocks(clip_model)[-unfreeze_last_n:]:
            for p in blk.parameters():
                p.requires_grad = True
    if train_layernorm:
        for _, m in clip_model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True



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
            raise ImportError("diffusers is required for reconstruction_residual")
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
            raise RuntimeError("Unexpected VAE encode output")
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
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.token_pool_size = token_pool_size
        self.reconstruction_use_fft_residual = reconstruction_use_fft_residual
        self.npr_scales = normalize_npr_scales(npr_scales)
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
        lap_k = self.laplacian_kernel.to(device=x.device, dtype=x.dtype)
        residuals = F.conv2d(x, lap_k, padding=1, groups=3)
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
            mag, phase = self.fft(x)
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
            wav28, wav14, wavelet_coeffs = self.wavelet_branch(x)
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


class WeightedFocalBCE(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0, alpha: Optional[float] = 0.65):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor([1.0]))
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        pos_weight = self.pos_weight.to(logits.device, logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal = focal * alpha_t

        loss = bce * focal

        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")


class AIDEStyleFusionDetector(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-L/14",
        clip_finetune_last_n: int = 0,
        clip_train_layernorm: bool = True,
        semantic_dropout: float = 0.10,
        semantic_token_mask_prob: float = SEMANTIC_ONLY_SEMANTIC_TOKEN_MASK_PROB_DEFAULT,
        patch_shuffle_prob: float = DEFAULT_PATCH_SHUFFLE_PROB,
        fusion_dim: int = 512,
        artifact_feature_dim: int = 192,
        artifact_token_dim: int = 128,
        token_pool_size: int = 14,
        artifact_branches: Optional[Sequence[str]] = None,
        reconstruction_vae_path: Optional[str] = None,
        reconstruction_vae_subfolder: Optional[str] = None,
        reconstruction_vae_dtype: str = "auto",
        reconstruction_use_fft_residual: bool = False,
        npr_scales: Optional[Sequence[float]] = None,
        fusion_mix_space: str = "clipped_logit",
        semantic_logit_clip: float = 5.0,
        artifact_logit_clip: float = 3.0,
        use_logit_temperature_calibration: bool = True,
        initial_semantic_temperature: float = 1.0,
        initial_artifact_temperature: float = 1.0,
        use_branch_logit_bias: bool = True,
        output_mix_mode: str = "learned_gate",
        fixed_artifact_prob_weight: float = 0.20,
        semantic_safe_semantic_prob_threshold: float = 0.50,
        semantic_safe_artifact_prob_threshold: float = 0.30,
        enable_artifact_failure_simulator: bool = False,
        artifact_failure_prob: float = 0.35,
        artifact_failure_bias_min: float = 4.0,
        artifact_failure_bias_max: float = 8.0,
        artifact_failure_mode: str = "opposite_semantic",
        artifact_failure_feature_dropout: float = 0.50,
        artifact_failure_attention_prob: float = 0.50,
        artifact_failure_attention_strength: float = 0.70,
        model_device: str = "cpu",
        use_clip_nn_fusion: bool = False,
        clip_nn_logit_clip: float = 5.0,
    ):
        super().__init__()
        self.model_device = model_device
        self.clip_model = load_clip_model(clip_model_name, device=model_device)
        self.feature_dim = self.clip_model.visual.output_dim
        self.semantic_dropout = float(semantic_dropout)
        self.semantic_token_mask_prob = float(semantic_token_mask_prob)
        self.patch_shuffle_prob = float(patch_shuffle_prob)
        self.fusion_dim = int(fusion_dim)
        self.use_clip_nn_fusion = bool(use_clip_nn_fusion)
        self.clip_nn_logit_clip = float(clip_nn_logit_clip)
        self.logit_feature_dim = 13 if self.use_clip_nn_fusion else 8
        self.reliability_feature_dim = 8
        self.fusion_input_dim = self.logit_feature_dim + self.reliability_feature_dim
        self.reliability_norm = nn.LayerNorm(self.reliability_feature_dim)
        self.reliability_dropout = nn.Dropout(0.05)

        self.fusion_mix_space = str(fusion_mix_space).lower().strip()
        if self.fusion_mix_space not in {"raw_logit", "clipped_logit", "probability"}:
            raise ValueError(f"Unsupported fusion_mix_space: {fusion_mix_space}")
        self.semantic_logit_clip = float(semantic_logit_clip)
        self.artifact_logit_clip = float(artifact_logit_clip)
        self.use_logit_temperature_calibration = bool(use_logit_temperature_calibration)
        self.use_branch_logit_bias = bool(use_branch_logit_bias)
        self.output_mix_mode = str(output_mix_mode).lower().strip()
        if self.output_mix_mode not in {"learned_gate", "fixed_prob", "learned_gate_with_semantic_safety", "fixed_prob_with_semantic_safety"}:
            raise ValueError(f"Unsupported output_mix_mode: {output_mix_mode}")
        self.fixed_artifact_prob_weight = float(fixed_artifact_prob_weight)
        if not (0.0 <= self.fixed_artifact_prob_weight <= 1.0):
            raise ValueError("--fixed_artifact_prob_weight must be in [0, 1]")
        self.semantic_safe_semantic_prob_threshold = float(semantic_safe_semantic_prob_threshold)
        self.semantic_safe_artifact_prob_threshold = float(semantic_safe_artifact_prob_threshold)
        self.enable_artifact_failure_simulator = bool(enable_artifact_failure_simulator)
        self.artifact_failure_prob = float(artifact_failure_prob)
        self.artifact_failure_bias_min = float(artifact_failure_bias_min)
        self.artifact_failure_bias_max = float(max(artifact_failure_bias_max, artifact_failure_bias_min))
        self.artifact_failure_mode = str(artifact_failure_mode).lower().strip()
        if self.artifact_failure_mode not in {"opposite_semantic", "random_sign", "mixed"}:
            raise ValueError(f"Unsupported artifact_failure_mode: {artifact_failure_mode}")
        self.artifact_failure_feature_dropout = float(artifact_failure_feature_dropout)
        self.artifact_failure_attention_prob = float(artifact_failure_attention_prob)
        self.artifact_failure_attention_strength = float(artifact_failure_attention_strength)
        sem_t0 = max(float(initial_semantic_temperature), 1e-3)
        art_t0 = max(float(initial_artifact_temperature), 1e-3)
        self.semantic_logit_temperature_log = nn.Parameter(
            torch.tensor(math.log(sem_t0), dtype=torch.float32),
            requires_grad=self.use_logit_temperature_calibration,
        )
        self.artifact_logit_temperature_log = nn.Parameter(
            torch.tensor(math.log(art_t0), dtype=torch.float32),
            requires_grad=self.use_logit_temperature_calibration,
        )
        self.semantic_logit_bias = nn.Parameter(
            torch.zeros(1, dtype=torch.float32),
            requires_grad=self.use_branch_logit_bias,
        )
        self.artifact_logit_bias = nn.Parameter(
            torch.zeros(1, dtype=torch.float32),
            requires_grad=self.use_branch_logit_bias,
        )

        if clip_finetune_last_n > 0:
            set_clip_finetune(self.clip_model, clip_finetune_last_n, clip_train_layernorm)
            self.clip_model = cast_clip_to_fp32_for_finetune(self.clip_model)
            self.clip_trainable = True
        else:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.clip_trainable = False

        self.artifact_extractor = MultiScaleLocalArtifactExtractor(
            feature_dim=artifact_feature_dim,
            token_dim=artifact_token_dim,
            token_pool_size=token_pool_size,
            active_branches=artifact_branches,
            reconstruction_vae_path=reconstruction_vae_path,
            reconstruction_vae_subfolder=reconstruction_vae_subfolder,
            reconstruction_vae_dtype=reconstruction_vae_dtype,
            reconstruction_use_fft_residual=reconstruction_use_fft_residual,
            npr_scales=npr_scales,
        )

        self.semantic_classifier = nn.Linear(self.feature_dim, 1)
        self.artifact_classifier = nn.Linear(artifact_feature_dim, 1)

        # Keep these attributes so checkpoint-loading utilities remain compatible.
        self.semantic_global_proj = nn.Identity()
        self.artifact_global_proj = nn.Identity()

        hidden_dim = max(int(fusion_dim), 64)
        bottleneck_dim = max(hidden_dim // 2, 32)
        reliability_hidden_dim = max(min(hidden_dim, 128), 64)
        reliability_proj_dim = max(bottleneck_dim, 32)

        self.logit_calibrator = nn.ModuleDict({
            "sem_rel_proj": nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, reliability_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(reliability_hidden_dim, reliability_proj_dim),
            ),
            "art_rel_proj": nn.Sequential(
                nn.LayerNorm(artifact_feature_dim),
                nn.Linear(artifact_feature_dim, reliability_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(reliability_hidden_dim, reliability_proj_dim),
            ),
            "gate": nn.Sequential(
                nn.LayerNorm(self.fusion_input_dim),
                nn.Linear(self.fusion_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim, 1),
            ),
            "bias": nn.Sequential(
                nn.LayerNorm(self.fusion_input_dim),
                nn.Linear(self.fusion_input_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(bottleneck_dim, 1),
            ),
        })
    def _encode_clip_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        visual = self.clip_model.visual
        dtype = visual.conv1.weight.dtype
        x = x.to(dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj
        cls_token = x[:, 0].float()
        patch_tokens = x[:, 1:].float()
        return cls_token, patch_tokens

    def _maybe_mask_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.semantic_token_mask_prob <= 0:
            return patch_tokens
        b, n, _ = patch_tokens.shape
        keep = (torch.rand(b, n, 1, device=patch_tokens.device) > self.semantic_token_mask_prob).float()
        denom = keep.mean(dim=1, keepdim=True).clamp(min=1e-6)
        return patch_tokens * keep / denom

    def _apply_image_patch_shuffle(self, x: torch.Tensor, grid_size: int = 14) -> torch.Tensor:
        if not self.training or self.patch_shuffle_prob <= 0:
            return x
        if torch.rand(1, device=x.device).item() > self.patch_shuffle_prob:
            return x
        b, c, h, w = x.shape
        if h % grid_size != 0 or w % grid_size != 0:
            return x
        patch_h, patch_w = h // grid_size, w // grid_size
        x_unfold = x.view(b, c, grid_size, patch_h, grid_size, patch_w)
        patches = x_unfold.permute(0, 2, 4, 1, 3, 5).reshape(b, grid_size * grid_size, c, patch_h, patch_w)
        shuffled_patches = torch.empty_like(patches)
        for i in range(b):
            idx = torch.randperm(grid_size * grid_size, device=x.device)
            shuffled_patches[i] = patches[i, idx]
        x_folded = shuffled_patches.view(b, grid_size, grid_size, c, patch_h, patch_w)
        return x_folded.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

    def encode_semantic(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.clip_trainable:
            return self._encode_clip_tokens(x)
        with torch.no_grad():
            return self._encode_clip_tokens(x)

    @staticmethod
    def _split_branch_inputs(x):
        if isinstance(x, dict):
            semantic_x = x.get("semantic")
            artifact_x = x.get("artifact", semantic_x)
            clip_nn_logit = x.get("clip_nn_logit", None)
            if semantic_x is None:
                raise KeyError("Dual-view input must contain key 'semantic'")
            return semantic_x, artifact_x, clip_nn_logit
        return x, x, None

    def _build_logit_features(
        self,
        semantic_logit: torch.Tensor,
        artifact_logit: torch.Tensor,
        clip_nn_logit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        semantic_logit = semantic_logit.view(-1, 1)
        artifact_logit = artifact_logit.view(-1, 1)
        sem_prob = torch.sigmoid(semantic_logit)
        art_prob = torch.sigmoid(artifact_logit)
        sem_margin = torch.abs(sem_prob - 0.5) * 2.0
        art_margin = torch.abs(art_prob - 0.5) * 2.0
        logit_diff = semantic_logit - artifact_logit
        abs_logit_diff = torch.abs(logit_diff)
        parts = [semantic_logit, artifact_logit, logit_diff, abs_logit_diff, sem_prob, art_prob, sem_margin, art_margin]
        if self.use_clip_nn_fusion:
            if clip_nn_logit is None:
                clip_nn_logit = torch.zeros_like(semantic_logit)
            clip_nn_logit = clip_nn_logit.view(-1, 1).to(device=semantic_logit.device, dtype=semantic_logit.dtype)
            if self.clip_nn_logit_clip > 0:
                clip_nn_logit = clip_nn_logit.clamp(-self.clip_nn_logit_clip, self.clip_nn_logit_clip)
            nn_prob = torch.sigmoid(clip_nn_logit)
            nn_margin = torch.abs(nn_prob - 0.5) * 2.0
            parts.extend([clip_nn_logit, nn_prob, semantic_logit - clip_nn_logit, artifact_logit - clip_nn_logit, nn_margin])
        return torch.cat(parts, dim=1)

    @staticmethod
    def _build_attention_reliability(artifact_attention: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        att = artifact_attention.float().clamp(min=1e-8)
        num_branches = int(att.shape[1])

        att_max = att.max(dim=1, keepdim=True).values
        if num_branches >= 2:
            top2 = torch.topk(att, k=2, dim=1).values
            att_gap = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
        else:
            att_gap = torch.zeros_like(att_max)

        denom = math.log(max(num_branches, 2))
        att_entropy = -(att * att.log()).sum(dim=1, keepdim=True) / denom
        att_std = att.std(dim=1, keepdim=True, unbiased=False)
        return att_max, att_gap, att_entropy, att_std

    def _build_reliability_features(
        self,
        semantic_features: torch.Tensor,
        artifact_features: torch.Tensor,
        artifact_attention: torch.Tensor,
    ) -> torch.Tensor:
        sem_feat = semantic_features.float()
        art_feat = artifact_features.float()

        sem_norm = torch.log1p(sem_feat.norm(dim=1, keepdim=True))
        art_norm = torch.log1p(art_feat.norm(dim=1, keepdim=True))

        sem_proj = self.logit_calibrator["sem_rel_proj"](sem_feat)
        art_proj = self.logit_calibrator["art_rel_proj"](art_feat)

        sem_proj_n = F.normalize(sem_proj, dim=1)
        art_proj_n = F.normalize(art_proj, dim=1)
        feat_cos = (sem_proj_n * art_proj_n).sum(dim=1, keepdim=True)
        feat_l2 = torch.log1p((sem_proj - art_proj).pow(2).sum(dim=1, keepdim=True).sqrt())

        att_max, att_gap, att_entropy, att_std = self._build_attention_reliability(artifact_attention)

        return torch.cat([
            sem_norm,
            art_norm,
            feat_cos,
            feat_l2,
            att_max,
            att_gap,
            att_entropy,
            att_std,
        ], dim=1)

    @staticmethod
    def _clamp_logit_for_mix(logit: torch.Tensor, clip_value: float) -> torch.Tensor:
        clip_value = float(clip_value)
        if clip_value <= 0:
            return logit
        return logit.clamp(min=-clip_value, max=clip_value)

    def _calibrate_branch_logits(
        self,
        semantic_logit: torch.Tensor,
        artifact_logit: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic_logit = semantic_logit.view(-1, 1).float()
        artifact_logit = artifact_logit.view(-1, 1).float()

        if self.use_logit_temperature_calibration:
            semantic_temperature = self.semantic_logit_temperature_log.float().exp().clamp(1e-3, 100.0)
            artifact_temperature = self.artifact_logit_temperature_log.float().exp().clamp(1e-3, 100.0)
        else:
            semantic_temperature = semantic_logit.new_tensor(1.0)
            artifact_temperature = artifact_logit.new_tensor(1.0)

        semantic_bias = self.semantic_logit_bias.to(device=semantic_logit.device, dtype=semantic_logit.dtype) if self.use_branch_logit_bias else semantic_logit.new_zeros(1)
        artifact_bias = self.artifact_logit_bias.to(device=artifact_logit.device, dtype=artifact_logit.dtype) if self.use_branch_logit_bias else artifact_logit.new_zeros(1)

        semantic_calibrated = semantic_logit / semantic_temperature.to(semantic_logit.device, semantic_logit.dtype) + semantic_bias
        artifact_calibrated = artifact_logit / artifact_temperature.to(artifact_logit.device, artifact_logit.dtype) + artifact_bias
        return semantic_calibrated, artifact_calibrated, semantic_temperature, artifact_temperature, semantic_bias, artifact_bias

    def _mix_branch_logits(
        self,
        fusion_gate: torch.Tensor,
        semantic_logit_calibrated: torch.Tensor,
        artifact_logit_calibrated: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mix_space = self.fusion_mix_space
        if mix_space == "raw_logit":
            semantic_for_mix = semantic_logit_calibrated
            artifact_for_mix = artifact_logit_calibrated
            mixed_logit = fusion_gate * artifact_for_mix + (1.0 - fusion_gate) * semantic_for_mix
            mixed_prob = torch.sigmoid(mixed_logit)
            return mixed_logit, semantic_for_mix, artifact_for_mix, mixed_prob

        semantic_for_mix = self._clamp_logit_for_mix(semantic_logit_calibrated, self.semantic_logit_clip)
        artifact_for_mix = self._clamp_logit_for_mix(artifact_logit_calibrated, self.artifact_logit_clip)

        if mix_space == "probability":
            semantic_prob = torch.sigmoid(semantic_for_mix)
            artifact_prob = torch.sigmoid(artifact_for_mix)
            mixed_prob = fusion_gate * artifact_prob + (1.0 - fusion_gate) * semantic_prob
            mixed_prob = mixed_prob.clamp(1e-6, 1.0 - 1e-6)
            mixed_logit = torch.logit(mixed_prob)
            return mixed_logit, semantic_for_mix, artifact_for_mix, mixed_prob

        mixed_logit = fusion_gate * artifact_for_mix + (1.0 - fusion_gate) * semantic_for_mix
        mixed_prob = torch.sigmoid(mixed_logit)
        return mixed_logit, semantic_for_mix, artifact_for_mix, mixed_prob

    def _simulate_artifact_failure(
        self,
        semantic_logit: torch.Tensor,
        artifact_logit: torch.Tensor,
        artifact_features: torch.Tensor,
        artifact_attention: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b = int(artifact_logit.view(-1, 1).shape[0])
        device = artifact_logit.device
        dtype = artifact_logit.dtype

        empty = artifact_logit.new_zeros(b, 1)
        sim_info = {
            "artifact_failure_mask": empty,
            "artifact_failure_logit_bias": empty,
            "artifact_failure_attention_mask": empty,
            "artifact_logit_for_fusion_raw": artifact_logit.view(-1, 1),
        }

        if (not self.training) or (not self.enable_artifact_failure_simulator) or self.artifact_failure_prob <= 0:
            return artifact_logit, artifact_features, artifact_attention, sim_info

        sample_mask = (torch.rand(b, 1, device=device) < self.artifact_failure_prob).float()
        if sample_mask.sum() <= 0:
            return artifact_logit, artifact_features, artifact_attention, sim_info

        mag_min = max(float(self.artifact_failure_bias_min), 0.0)
        mag_max = max(float(self.artifact_failure_bias_max), mag_min)
        magnitude = mag_min + (mag_max - mag_min) * torch.rand(b, 1, device=device, dtype=dtype)

        sem_fake = (torch.sigmoid(semantic_logit.detach().view(-1, 1)) > 0.5).float()
        opposite_sign = torch.where(sem_fake > 0.5, -torch.ones_like(magnitude), torch.ones_like(magnitude))
        random_sign = torch.where(
            torch.rand(b, 1, device=device) < 0.5,
            -torch.ones_like(magnitude),
            torch.ones_like(magnitude),
        )
        mode = self.artifact_failure_mode
        if mode == "opposite_semantic":
            sign = opposite_sign
        elif mode == "random_sign":
            sign = random_sign
        else:
            use_opposite = (torch.rand(b, 1, device=device) < 0.7).float()
            sign = use_opposite * opposite_sign + (1.0 - use_opposite) * random_sign

        logit_bias = sign * magnitude * sample_mask.to(dtype)
        corrupted_artifact_logit = artifact_logit.view(-1, 1) + logit_bias

        corrupted_features = artifact_features
        p_drop = float(self.artifact_failure_feature_dropout)
        if p_drop > 0:
            dropped = F.dropout(artifact_features, p=min(max(p_drop, 0.0), 0.95), training=True)
            corrupted_features = torch.where(sample_mask.bool(), dropped, artifact_features)

        corrupted_attention = artifact_attention
        att_prob = float(self.artifact_failure_attention_prob)
        if artifact_attention is not None and artifact_attention.ndim == 2 and att_prob > 0:
            att_mask = ((torch.rand(b, 1, device=device) < att_prob).float() * sample_mask).to(artifact_attention.dtype)
            strength = min(max(float(self.artifact_failure_attention_strength), 0.0), 1.0)
            uniform = torch.ones_like(artifact_attention) / max(int(artifact_attention.shape[1]), 1)
            noise = torch.rand_like(artifact_attention)
            noise = noise / noise.sum(dim=1, keepdim=True).clamp(min=1e-6)
            perturbed = (1.0 - strength) * artifact_attention + strength * (0.5 * uniform + 0.5 * noise)
            perturbed = perturbed / perturbed.sum(dim=1, keepdim=True).clamp(min=1e-6)
            corrupted_attention = torch.where(att_mask.bool(), perturbed, artifact_attention)
            sim_info["artifact_failure_attention_mask"] = att_mask.float()

        sim_info.update({
            "artifact_failure_mask": sample_mask.float(),
            "artifact_failure_logit_bias": logit_bias.float(),
            "artifact_logit_for_fusion_raw": corrupted_artifact_logit.detach().float(),
        })
        return corrupted_artifact_logit, corrupted_features, corrupted_attention, sim_info

    def _run_joint_fusion(
        self,
        semantic_features: torch.Tensor,
        artifact_features: torch.Tensor,
        artifact_attention: torch.Tensor,
        semantic_logit: torch.Tensor,
        artifact_logit: torch.Tensor,
        clip_nn_logit: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logit_features = self._build_logit_features(semantic_logit, artifact_logit, clip_nn_logit=clip_nn_logit)
        raw_reliability_features = self._build_reliability_features(
            semantic_features,
            artifact_features,
            artifact_attention,
        )
        reliability_features = self.reliability_norm(raw_reliability_features)
        if self.training:
            reliability_features = self.reliability_dropout(reliability_features)
        calibrator_input = torch.cat([logit_features, reliability_features], dim=1)

        fusion_gate = torch.sigmoid(self.logit_calibrator["gate"](calibrator_input))
        fusion_bias = self.logit_calibrator["bias"](calibrator_input)

        (
            semantic_logit_calibrated,
            artifact_logit_calibrated,
            semantic_temperature,
            artifact_temperature,
            semantic_branch_bias,
            artifact_branch_bias,
        ) = self._calibrate_branch_logits(semantic_logit, artifact_logit)
        mixed_logit, semantic_logit_for_mix, artifact_logit_for_mix, mixed_prob = self._mix_branch_logits(
            fusion_gate=fusion_gate,
            semantic_logit_calibrated=semantic_logit_calibrated,
            artifact_logit_calibrated=artifact_logit_calibrated,
        )
        learned_final_logit = mixed_logit + fusion_bias

        semantic_prob_for_mix = torch.sigmoid(semantic_logit_for_mix)
        artifact_prob_for_mix = torch.sigmoid(artifact_logit_for_mix)
        learned_final_prob = torch.sigmoid(learned_final_logit).clamp(1e-6, 1.0 - 1e-6)

        output_mix_mode = self.output_mix_mode
        fixed_w = float(self.fixed_artifact_prob_weight)
        fixed_prob = ((1.0 - fixed_w) * semantic_prob_for_mix + fixed_w * artifact_prob_for_mix).clamp(1e-6, 1.0 - 1e-6)

        if output_mix_mode.startswith("fixed_prob"):
            final_prob = fixed_prob
        else:
            final_prob = learned_final_prob

        semantic_safe_mask = (
            (semantic_prob_for_mix > self.semantic_safe_semantic_prob_threshold)
            & (artifact_prob_for_mix < self.semantic_safe_artifact_prob_threshold)
        )
        if output_mix_mode.endswith("semantic_safety"):
            final_prob = torch.where(semantic_safe_mask, semantic_prob_for_mix, final_prob).clamp(1e-6, 1.0 - 1e-6)

        final_logit = torch.logit(final_prob)

        return {
            "final_output": final_logit,
            "calibrator_input": calibrator_input,
            "logit_features": logit_features,
            "raw_reliability_features": raw_reliability_features,
            "reliability_features": reliability_features,
            "fusion_gate": fusion_gate,
            "fusion_bias": fusion_bias,
            "mixed_logit": mixed_logit,
            "mixed_prob": mixed_prob,
            "learned_final_logit": learned_final_logit,
            "learned_final_prob": learned_final_prob,
            "fixed_prob": fixed_prob,
            "semantic_safe_mask": semantic_safe_mask.float(),
            "output_mix_mode_tensor": final_logit.new_tensor(float({"learned_gate": 0, "fixed_prob": 1, "learned_gate_with_semantic_safety": 2, "fixed_prob_with_semantic_safety": 3}[self.output_mix_mode])),
            "semantic_logit_calibrated": semantic_logit_calibrated,
            "artifact_logit_calibrated": artifact_logit_calibrated,
            "semantic_logit_for_mix": semantic_logit_for_mix,
            "artifact_logit_for_mix": artifact_logit_for_mix,
            "semantic_temperature": semantic_temperature.view(1),
            "artifact_temperature": artifact_temperature.view(1),
            "semantic_branch_bias": semantic_branch_bias.view(1),
            "artifact_branch_bias": artifact_branch_bias.view(1),
            "clip_nn_logit": (clip_nn_logit.view(-1, 1).to(final_logit.device, final_logit.dtype) if clip_nn_logit is not None else torch.full_like(final_logit, float("nan"))),
            "clip_nn_prob": (torch.sigmoid(clip_nn_logit.view(-1, 1).to(final_logit.device, final_logit.dtype)) if clip_nn_logit is not None else torch.full_like(final_logit, float("nan"))),
        }
    def forward(self, x, return_details: bool = False):
        semantic_x, artifact_x, clip_nn_logit = self._split_branch_inputs(x)
        semantic_x = self._apply_image_patch_shuffle(semantic_x)
        semantic_features, semantic_tokens = self.encode_semantic(semantic_x)
        semantic_tokens = self._maybe_mask_tokens(semantic_tokens)
        if self.training and self.semantic_dropout > 0:
            semantic_features = F.dropout(semantic_features, p=self.semantic_dropout, training=True)
        artifact_features, artifact_tokens, artifact_attention = self.artifact_extractor(artifact_x)
        if self.training:
            artifact_features = F.dropout(artifact_features, p=self.artifact_feature_dropout, training=True)

        semantic_output = self.semantic_classifier(semantic_features)
        artifact_output = self.artifact_classifier(artifact_features)
        artifact_output_for_fusion, artifact_features_for_fusion, artifact_attention_for_fusion, artifact_failure_info = self._simulate_artifact_failure(
            semantic_logit=semantic_output,
            artifact_logit=artifact_output,
            artifact_features=artifact_features,
            artifact_attention=artifact_attention,
        )
        fusion = self._run_joint_fusion(
            semantic_features,
            artifact_features_for_fusion,
            artifact_attention_for_fusion,
            semantic_output,
            artifact_output_for_fusion,
            clip_nn_logit=clip_nn_logit,
        )
        fusion.update(artifact_failure_info)
        fusion["artifact_output_for_fusion"] = artifact_output_for_fusion
        final_output = fusion["final_output"]

        if not return_details:
            return final_output
        return {
            "final_output": final_output,
            "semantic_output": semantic_output,
            "artifact_output": artifact_output,
            "semantic_features": semantic_features,
            "artifact_features": artifact_features,
            "semantic_tokens": semantic_tokens,
            "artifact_tokens": artifact_tokens,
            "artifact_attention": artifact_attention,
            **fusion,
        }


class LogitCalibratorLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        final_weight: float = 1.0,
        semantic_aux_weight: float = 0.0,
        artifact_aux_weight: float = 0.0,
        pos_scale: float = 1.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.65,
        macro_loss_weight: float = 0.75,
        sample_loss_weight: float = 0.25,
        gate_reg_weight: float = 0.0,
        bias_reg_weight: float = 0.002,
        regret_weight: float = 0.0,
        branch_consistency_weight: float = 0.0,
        router_supervision_weight: float = 0.0,
        router_target_mode: str = "model_macroacc",
        regret_target_mode: str = "model_macroacc",
        artifact_failure_gate_weight: float = 0.0,
        artifact_failure_consistency_weight: float = 0.0,
        disagreement_safe_weight: float = 0.0,
        disagreement_margin: float = 0.30,
    ):
        super().__init__()
        self.final_weight = float(final_weight)
        self.semantic_aux_weight = float(semantic_aux_weight)
        self.artifact_aux_weight = float(artifact_aux_weight)
        self.macro_loss_weight = float(macro_loss_weight)
        self.sample_loss_weight = float(sample_loss_weight)
        self.gate_reg_weight = float(gate_reg_weight)
        self.bias_reg_weight = float(bias_reg_weight)
        self.regret_weight = float(regret_weight)
        self.branch_consistency_weight = float(branch_consistency_weight)
        self.router_supervision_weight = float(router_supervision_weight)
        self.router_target_mode = str(router_target_mode)
        self.regret_target_mode = str(regret_target_mode)
        self.artifact_failure_gate_weight = float(artifact_failure_gate_weight)
        self.artifact_failure_consistency_weight = float(artifact_failure_consistency_weight)
        self.disagreement_safe_weight = float(disagreement_safe_weight)
        self.disagreement_margin = float(disagreement_margin)

        if class_weights is not None and class_weights.numel() >= 2:
            pos_weight = (class_weights[1] / class_weights[0]).clamp(min=1e-6) * pos_scale
        else:
            pos_weight = torch.tensor([pos_scale], dtype=torch.float32)

        self.loss_fn = WeightedFocalBCE(
            pos_weight=pos_weight.float(),
            gamma=focal_gamma,
            alpha=focal_alpha,
        )

    @staticmethod
    def _normalize_model_names(model_names: Optional[List[str]]) -> Optional[List[str]]:
        if model_names is None:
            return None
        return [str(v).replace("\\", "/").split("/")[0] for v in model_names]

    def _group_mean(self, per_sample_loss: torch.Tensor, model_names: Optional[List[str]]) -> torch.Tensor:
        if model_names is None or len(model_names) == 0:
            return per_sample_loss.mean()

        model_names = self._normalize_model_names(model_names)
        grouped_indices = defaultdict(list)
        for idx, model_name in enumerate(model_names):
            grouped_indices[model_name].append(idx)

        group_losses = []
        for _, idxs in grouped_indices.items():
            idx_tensor = torch.as_tensor(idxs, dtype=torch.long, device=per_sample_loss.device)
            group_losses.append(per_sample_loss.index_select(0, idx_tensor).mean())

        return torch.stack(group_losses).mean() if group_losses else per_sample_loss.mean()

    @staticmethod
    def _prob_bce_no_autocast(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Manual BCE on probabilities. Avoid F.binary_cross_entropy because PyTorch forbids it under AMP/autocast.
        probs_f = probs.float().clamp(1e-6, 1.0 - 1e-6)
        targets_f = targets.float()
        return -(targets_f * torch.log(probs_f) + (1.0 - targets_f) * torch.log1p(-probs_f))

    def _build_sample_bce_router_target(
        self,
        semantic_per_sample: torch.Tensor,
        artifact_per_sample: torch.Tensor,
    ) -> torch.Tensor:
        # gate=1 means artifact, gate=0 means semantic.
        return (artifact_per_sample.detach() < semantic_per_sample.detach()).float()

    def _build_model_macroacc_router_target(
        self,
        semantic_logit: torch.Tensor,
        artifact_logit: torch.Tensor,
        targets: torch.Tensor,
        model_names: Optional[List[str]],
    ) -> torch.Tensor:
        if model_names is None or len(model_names) == 0:
            sem_bce = F.binary_cross_entropy_with_logits(semantic_logit, targets, reduction="none")
            art_bce = F.binary_cross_entropy_with_logits(artifact_logit, targets, reduction="none")
            return self._build_sample_bce_router_target(sem_bce, art_bce)

        model_names = self._normalize_model_names(model_names)
        sem_pred = (torch.sigmoid(semantic_logit.detach()) > 0.5).float()
        art_pred = (torch.sigmoid(artifact_logit.detach()) > 0.5).float()
        y = targets.float()
        grouped_indices = defaultdict(list)
        for idx, model_name in enumerate(model_names):
            grouped_indices[model_name].append(idx)

        target = torch.zeros_like(y, dtype=torch.float32)
        for _, idxs in grouped_indices.items():
            idx_tensor = torch.as_tensor(idxs, dtype=torch.long, device=y.device)
            sem_acc = (sem_pred.index_select(0, idx_tensor) == y.index_select(0, idx_tensor)).float().mean()
            art_acc = (art_pred.index_select(0, idx_tensor) == y.index_select(0, idx_tensor)).float().mean()
            # MacroAccuracy is an equal-weighted mean over model accuracies.
            # Therefore every sample from the same model receives the branch target
            # of the branch with higher fixed-0.5 accuracy for that model in this batch.
            branch_target = 1.0 if float(art_acc.item()) > float(sem_acc.item()) else 0.0
            target.index_fill_(0, idx_tensor, branch_target)
        return target

    def _build_router_target(
        self,
        semantic_logit: torch.Tensor,
        artifact_logit: torch.Tensor,
        targets: torch.Tensor,
        semantic_per_sample: torch.Tensor,
        artifact_per_sample: torch.Tensor,
        model_names: Optional[List[str]],
    ) -> torch.Tensor:
        mode = self.router_target_mode.lower()
        if mode in {"model_macroacc", "macroacc", "model_accuracy"}:
            return self._build_model_macroacc_router_target(semantic_logit, artifact_logit, targets, model_names)
        if mode in {"sample_bce", "bce", "sample_loss"}:
            return self._build_sample_bce_router_target(semantic_per_sample, artifact_per_sample)
        raise ValueError(f"Unsupported router_target_mode: {self.router_target_mode}")

    def _branch_consistency_loss(
        self,
        final_logit: torch.Tensor,
        semantic_logit: torch.Tensor,
        artifact_logit: torch.Tensor,
        router_target: torch.Tensor,
        model_names: Optional[List[str]],
    ) -> torch.Tensor:
        # Align final probability with the branch selected by the macroAcc-oriented router target.
        # gate/router target: 1 means artifact, 0 means semantic. Branch probabilities are detached
        # so this auxiliary term trains only the fusion/calibration path, not the frozen branches.
        semantic_prob = torch.sigmoid(semantic_logit.detach())
        artifact_prob = torch.sigmoid(artifact_logit.detach())
        target_prob = torch.where(
            router_target.detach() > 0.5,
            artifact_prob,
            semantic_prob,
        ).detach().to(dtype=final_logit.dtype)
        # Use logits-version BCE. F.binary_cross_entropy on probabilities is unsafe under AMP/autocast.
        # BCEWithLogits accepts soft targets in [0, 1] and keeps the same branch-consistency objective.
        per_sample = F.binary_cross_entropy_with_logits(final_logit, target_prob, reduction="none")
        group_loss = self._group_mean(per_sample, model_names)
        sample_loss = per_sample.mean()
        return self.macro_loss_weight * group_loss + self.sample_loss_weight * sample_loss

    def _macro_regret_loss(
        self,
        final_per_sample: torch.Tensor,
        selected_branch_per_sample: torch.Tensor,
        model_names: Optional[List[str]],
    ) -> torch.Tensor:
        if model_names is None or len(model_names) == 0:
            return F.relu(final_per_sample.mean() - selected_branch_per_sample.mean())
        model_names = self._normalize_model_names(model_names)
        grouped_indices = defaultdict(list)
        for idx, model_name in enumerate(model_names):
            grouped_indices[model_name].append(idx)
        regrets = []
        for _, idxs in grouped_indices.items():
            idx_tensor = torch.as_tensor(idxs, dtype=torch.long, device=final_per_sample.device)
            final_group = final_per_sample.index_select(0, idx_tensor).mean()
            branch_group = selected_branch_per_sample.index_select(0, idx_tensor).mean()
            regrets.append(F.relu(final_group - branch_group))
        return torch.stack(regrets).mean() if regrets else F.relu(final_per_sample.mean() - selected_branch_per_sample.mean())

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        model_names: Optional[List[str]] = None,
    ):
        targets = targets.float().view(-1)

        final_logit = outputs["final_output"].view(-1)
        semantic_logit = outputs["semantic_output"].view(-1)
        artifact_logit = outputs["artifact_output"].view(-1)

        final_per_sample = self.loss_fn(final_logit, targets, reduction="none")
        semantic_per_sample = self.loss_fn(semantic_logit, targets, reduction="none")
        artifact_per_sample = self.loss_fn(artifact_logit, targets, reduction="none")

        final_group_loss = self._group_mean(final_per_sample, model_names)
        final_sample_loss = final_per_sample.mean()
        final_loss = self.macro_loss_weight * final_group_loss + self.sample_loss_weight * final_sample_loss

        semantic_loss = semantic_per_sample.mean()
        artifact_loss = artifact_per_sample.mean()

        total = (
            self.final_weight * final_loss
            + self.semantic_aux_weight * semantic_loss
            + self.artifact_aux_weight * artifact_loss
        )

        router_target = self._build_router_target(
            semantic_logit=semantic_logit,
            artifact_logit=artifact_logit,
            targets=targets,
            semantic_per_sample=semantic_per_sample,
            artifact_per_sample=artifact_per_sample,
            model_names=model_names,
        )

        if self.regret_target_mode.lower() in {"model_macroacc", "macroacc", "model_accuracy"}:
            selected_branch_per_sample = torch.where(
                router_target.detach() > 0.5,
                artifact_per_sample.detach(),
                semantic_per_sample.detach(),
            )
            regret_loss = self._macro_regret_loss(final_per_sample, selected_branch_per_sample, model_names)
        else:
            best_branch_per_sample = torch.minimum(semantic_per_sample, artifact_per_sample).detach()
            regret_per_sample = F.relu(final_per_sample - best_branch_per_sample)
            regret_group_loss = self._group_mean(regret_per_sample, model_names)
            regret_sample_loss = regret_per_sample.mean()
            regret_loss = self.macro_loss_weight * regret_group_loss + self.sample_loss_weight * regret_sample_loss

        if self.regret_weight > 0:
            total = total + self.regret_weight * regret_loss

        branch_consistency_loss = torch.tensor(0.0, device=final_logit.device)
        if self.branch_consistency_weight > 0:
            branch_consistency_loss = self._branch_consistency_loss(
                final_logit=final_logit,
                semantic_logit=semantic_logit,
                artifact_logit=artifact_logit,
                router_target=router_target,
                model_names=model_names,
            )
            total = total + self.branch_consistency_weight * branch_consistency_loss

        router_loss = torch.tensor(0.0, device=final_logit.device)
        router_target_mean = router_target.mean()
        gate_reg = torch.tensor(0.0, device=final_logit.device)
        if "fusion_gate" in outputs:
            gate = outputs["fusion_gate"].view(-1).clamp(1e-6, 1.0 - 1e-6)
            gate_reg = (gate.mean() - 0.5).abs()
            total = total + self.gate_reg_weight * gate_reg

            if self.router_supervision_weight > 0:
                router_per_sample = self._prob_bce_no_autocast(gate, router_target.detach())
                router_group_loss = self._group_mean(router_per_sample, model_names)
                router_sample_loss = router_per_sample.mean()
                router_loss = self.macro_loss_weight * router_group_loss + self.sample_loss_weight * router_sample_loss
                total = total + self.router_supervision_weight * router_loss

        artifact_failure_gate_loss = torch.tensor(0.0, device=final_logit.device)
        artifact_failure_consistency_loss = torch.tensor(0.0, device=final_logit.device)
        disagreement_safe_loss = torch.tensor(0.0, device=final_logit.device)

        if "fusion_gate" in outputs:
            gate_for_aux = outputs["fusion_gate"].view(-1).clamp(1e-6, 1.0 - 1e-6)

            failure_mask = outputs.get("artifact_failure_mask", None)
            if failure_mask is not None:
                failure_mask = failure_mask.view(-1).float()
                if failure_mask.sum() > 0 and self.artifact_failure_gate_weight > 0:
                    artifact_failure_gate_loss = self._prob_bce_no_autocast(
                        gate_for_aux,
                        torch.zeros_like(gate_for_aux),
                    )
                    artifact_failure_gate_loss = (artifact_failure_gate_loss * failure_mask).sum() / failure_mask.sum().clamp(min=1.0)
                    total = total + self.artifact_failure_gate_weight * artifact_failure_gate_loss

                if failure_mask.sum() > 0 and self.artifact_failure_consistency_weight > 0:
                    final_prob_for_aux = torch.sigmoid(final_logit)
                    semantic_prob_for_aux = torch.sigmoid(semantic_logit.detach())
                    artifact_failure_consistency_loss = (
                        (final_prob_for_aux - semantic_prob_for_aux).pow(2) * failure_mask
                    ).sum() / failure_mask.sum().clamp(min=1.0)
                    total = total + self.artifact_failure_consistency_weight * artifact_failure_consistency_loss

            if self.disagreement_safe_weight > 0:
                sem_prob = torch.sigmoid(semantic_logit.detach())
                art_prob = torch.sigmoid(artifact_logit.detach())
                sem_pred = (sem_prob > 0.5).float()
                art_pred = (art_prob > 0.5).float()
                semantic_correct = (sem_pred == targets).float()
                artifact_wrong = (art_pred != targets).float()
                disagreement = (sem_prob - art_prob).abs() > self.disagreement_margin
                safe_mask = (semantic_correct * artifact_wrong * disagreement.float()).float()
                if safe_mask.sum() > 0:
                    per_sample = self._prob_bce_no_autocast(
                        gate_for_aux,
                        torch.zeros_like(gate_for_aux),
                    )
                    disagreement_safe_loss = (per_sample * safe_mask).sum() / safe_mask.sum().clamp(min=1.0)
                    total = total + self.disagreement_safe_weight * disagreement_safe_loss

        bias_reg = torch.tensor(0.0, device=final_logit.device)
        if "fusion_bias" in outputs:
            bias_reg = outputs["fusion_bias"].view(-1).pow(2).mean()
            total = total + self.bias_reg_weight * bias_reg

        return total, {
            "total_loss": float(total.item()),
            "final_loss": float(final_loss.item()),
            "final_group_loss": float(final_group_loss.item()),
            "final_sample_loss": float(final_sample_loss.item()),
            "semantic_loss": float(semantic_loss.item()),
            "artifact_loss": float(artifact_loss.item()),
            "regret_loss": float(regret_loss.item()),
            "branch_consistency_loss": float(branch_consistency_loss.item()),
            "router_loss": float(router_loss.item()),
            "router_target_mean": float(router_target_mean.item()),
            "router_target_mode": self.router_target_mode,
            "regret_target_mode": self.regret_target_mode,
            "artifact_failure_gate_loss": float(artifact_failure_gate_loss.item()),
            "artifact_failure_consistency_loss": float(artifact_failure_consistency_loss.item()),
            "disagreement_safe_loss": float(disagreement_safe_loss.item()),
            "gate_reg": float(gate_reg.item()),
            "bias_reg": float(bias_reg.item()),
        }
def compute_class_weights_from_labels(labels: Sequence[int]) -> torch.Tensor:
    t = torch.tensor([int(v) for v in labels], dtype=torch.long)
    counts = torch.bincount(t, minlength=2).float().clamp(min=1.0)
    return 1.0 / counts




def _canonical_model_name(name: str) -> str:
    return str(name).replace("\\", "/").split("/")[0]


def build_model_balanced_sampler(dataset: ImageForgeryDataset) -> WeightedRandomSampler:
    model_counts = defaultdict(int)
    model_label_counts = defaultdict(int)

    for model_name, label in zip(dataset.models, dataset.labels):
        m = _canonical_model_name(model_name)
        y = int(label)
        model_counts[m] += 1
        model_label_counts[(m, y)] += 1

    weights = []
    for model_name, label in zip(dataset.models, dataset.labels):
        m = _canonical_model_name(model_name)
        y = int(label)

        w_model = 1.0 / max(model_counts[m], 1)
        w_model_label = 1.0 / max(model_label_counts[(m, y)], 1)

        w = math.sqrt(w_model * w_model_label)
        weights.append(w)

    weights = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
def build_optimizer(model: AIDEStyleFusionDetector, clip_lr: float, base_lr: float) -> optim.Optimizer:
    clip_params = [p for p in model.clip_model.parameters() if p.requires_grad]
    semantic_params = [p for p in model.semantic_classifier.parameters() if p.requires_grad]
    artifact_params = [p for p in list(model.artifact_extractor.parameters()) + list(model.artifact_classifier.parameters()) if p.requires_grad]
    calibrator_params = [p for p in model.logit_calibrator.parameters() if p.requires_grad]
    calibrator_params += [p for p in model.reliability_norm.parameters() if p.requires_grad]
    for name in (
        "semantic_logit_temperature_log",
        "artifact_logit_temperature_log",
        "semantic_logit_bias",
        "artifact_logit_bias",
    ):
        p = getattr(model, name, None)
        if p is not None and getattr(p, "requires_grad", False):
            calibrator_params.append(p)
    param_groups = []
    if artifact_params:
        param_groups.append({"params": artifact_params, "lr": base_lr * 0.5})
    if semantic_params:
        param_groups.append({"params": semantic_params, "lr": base_lr * 0.5})
    if calibrator_params:
        param_groups.append({"params": calibrator_params, "lr": base_lr})
    if clip_params:
        param_groups.append({"params": clip_params, "lr": clip_lr})
    return optim.AdamW(param_groups, weight_decay=0.02)



def unpack_fusion_batch(batch, device: str):
    if len(batch) == 4:
        inputs, labels, categories, clip_nn_logits = batch
        clip_nn_logits = clip_nn_logits.to(device, non_blocking=True).float().view(-1, 1)
    elif len(batch) == 3:
        inputs, labels, categories = batch
        clip_nn_logits = None
    else:
        raise ValueError(f"Unexpected batch format with {len(batch)} fields")

    labels = labels.to(device, non_blocking=True).float()
    if isinstance(inputs, dict):
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        if clip_nn_logits is not None:
            inputs["clip_nn_logit"] = clip_nn_logits
    else:
        img = inputs.to(device, non_blocking=True)
        inputs = {"semantic": img, "artifact": img, "clip_nn_logit": clip_nn_logits} if clip_nn_logits is not None else img
    return inputs, labels, categories


def compute_basic_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    preds = (probs > threshold).astype(np.int64)
    acc = float((preds == labels).mean()) if len(labels) > 0 else float("nan")
    real_mask = labels == 0
    fake_mask = labels == 1
    real_acc = float((preds[real_mask] == 0).mean()) if real_mask.any() else float("nan")
    fake_acc = float((preds[fake_mask] == 1).mean()) if fake_mask.any() else float("nan")
    balanced_acc = float(np.nanmean([real_acc, fake_acc]))
    ap = float("nan")
    if sk_average_precision_score is not None and len(np.unique(labels)) > 1:
        try:
            ap = float(sk_average_precision_score(labels, probs))
        except Exception:
            ap = float("nan")
    return {
        "accuracy": acc,
        "balanced_acc": balanced_acc,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "ap": ap,
    }

@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion: nn.Module, device: str, use_amp: bool, amp_dtype: str, desc: str = "Eval", show_progress: bool = True):
    model.eval()
    total_loss = 0.0
    total_n = 0
    final_probs, sem_probs, art_probs, labels_all, model_names_all = [], [], [], [], []

    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if show_progress else loader
    for batch in iterator:
        if batch is None:
            continue

        inputs, labels, categories = unpack_fusion_batch(batch, device)
        batch_model_names = [str(v).replace("\\", "/").split("/")[0] for v in categories]

        if use_amp and device.startswith("cuda"):
            dtype = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(inputs, return_details=True)
                loss, _ = criterion(outputs, labels, model_names=batch_model_names)
        else:
            outputs = model(inputs, return_details=True)
            loss, _ = criterion(outputs, labels, model_names=batch_model_names)

        total_loss += float(loss.item()) * labels.numel()
        total_n += int(labels.numel())

        labels_all.append(labels.detach().cpu().numpy())
        final_probs.append(torch.sigmoid(outputs["final_output"].view(-1)).detach().float().cpu().numpy())
        sem_probs.append(torch.sigmoid(outputs["semantic_output"].view(-1)).detach().float().cpu().numpy())
        art_probs.append(torch.sigmoid(outputs["artifact_output"].view(-1)).detach().float().cpu().numpy())
        model_names_all.extend(batch_model_names)

    labels_np = np.concatenate(labels_all) if labels_all else np.asarray([], dtype=np.int64)
    final_np = np.concatenate(final_probs) if final_probs else np.asarray([], dtype=np.float32)
    sem_np = np.concatenate(sem_probs) if sem_probs else np.asarray([], dtype=np.float32)
    art_np = np.concatenate(art_probs) if art_probs else np.asarray([], dtype=np.float32)

    return {
        "loss": total_loss / max(total_n, 1),
        "final": compute_threshold_metrics(labels_np, final_np, 0.5, model_names_all),
        "semantic": compute_threshold_metrics(labels_np, sem_np, 0.5, model_names_all),
        "artifact": compute_threshold_metrics(labels_np, art_np, 0.5, model_names_all),
    }
def train_one_epoch(model, loader, criterion, optimizer, device: str, use_amp: bool, amp_dtype: str, scaler, grad_clip: float, epoch: int = 1, total_epochs: int = 1):
    model.train()
    total_loss = 0.0
    total_n = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} - Train', leave=False, dynamic_ncols=True)

    for batch in pbar:
        if batch is None:
            continue

        inputs, labels, categories = unpack_fusion_batch(batch, device)
        batch_model_names = [str(v).replace("\\", "/").split("/")[0] for v in categories]

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.startswith("cuda"):
            dtype = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(inputs, return_details=True)
                loss, loss_info = criterion(outputs, labels, model_names=batch_model_names)
        else:
            outputs = model(inputs, return_details=True)
            loss, loss_info = criterion(outputs, labels, model_names=batch_model_names)

        if not torch.isfinite(loss):
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item()) * labels.numel()
        total_n += int(labels.numel())
        running_loss = total_loss / max(total_n, 1)
        pbar.set_postfix(
            loss=f"{running_loss:.4f}",
            regret=f"{loss_info.get('regret_loss', 0.0):.4f}",
            bc=f"{loss_info.get('branch_consistency_loss', 0.0):.4f}",
            router=f"{loss_info.get('router_loss', 0.0):.4f}",
        )

    return total_loss / max(total_n, 1)
def save_checkpoint(model, optimizer, epoch: int, best_metric: float, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)



def load_checkpoint(model, path: str, device: str, optimizer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass
    return ckpt



def save_metrics_json(metrics: Dict[str, object], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def build_epoch_checkpoint_path(base_path: str, epoch: int) -> str:
    base = Path(base_path)
    return str(base.with_name(f"{base.stem}_epoch_{epoch:03d}{base.suffix}"))



def _infer_artifact_branches_from_state_dict(state_dict: Dict[str, torch.Tensor], all_branches: Sequence[str]) -> List[str]:
    branch_key_map = {
        "high_freq": "high_freq_branch.",
        "compression": "compression_branch.",
        "spectral_mag": "spectral_mag_branch.",
        "spectral_phase": "spectral_phase_branch.",
        "wavelet": "wavelet_branch.",
        "npr": "npr_branch.",
        "reconstruction_residual": "reconstruction_branch.",
    }
    inferred = []
    for name in all_branches:
        prefix = branch_key_map[name]
        if any(str(k).startswith(prefix) for k in state_dict.keys()):
            inferred.append(name)
    return inferred


def inspect_artifact_checkpoint_build_meta(checkpoint_path: str, all_branches: Sequence[str]) -> Dict[str, object]:
    if not checkpoint_path:
        return {}
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Artifact checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return {}

    artifact_state = None
    artifact_branches = ckpt.get("artifact_branches", None)
    release_meta = ckpt.get("metadata", {}) if isinstance(ckpt.get("metadata", {}), dict) else {}
    artifact_build_meta = ckpt.get("artifact_build_meta", {}) if isinstance(ckpt.get("artifact_build_meta", {}), dict) else {}
    if not artifact_build_meta and isinstance(release_meta.get("artifact_build_meta", {}), dict):
        artifact_build_meta = release_meta.get("artifact_build_meta", {})
    if artifact_branches is None:
        artifact_branches = release_meta.get("artifact_branches", None)
    if artifact_branches is None:
        artifact_branches = artifact_build_meta.get("artifact_branches", None)
    checkpoint_type = ckpt.get("checkpoint_type", "")

    if checkpoint_type in {"artifact", "artifact_inference"} and isinstance(ckpt.get("artifact_extractor_state_dict"), dict):
        artifact_state = ckpt["artifact_extractor_state_dict"]
    elif isinstance(ckpt.get("model_state_dict"), dict):
        artifact_state = {
            key[len("artifact_extractor."):]: value
            for key, value in ckpt["model_state_dict"].items()
            if str(key).startswith("artifact_extractor.")
        }
    else:
        artifact_state = {
            key[len("artifact_extractor."):]: value
            for key, value in ckpt.items()
            if isinstance(key, str) and key.startswith("artifact_extractor.")
        }

    if artifact_branches is None and artifact_state:
        artifact_branches = _infer_artifact_branches_from_state_dict(artifact_state, all_branches)

    recon_weight = None
    if artifact_state:
        recon_weight = artifact_state.get("reconstruction_branch.stem.block.0.weight", None)

    reconstruction_in_channels = None
    reconstruction_use_fft_residual = False
    if torch.is_tensor(recon_weight) and recon_weight.ndim >= 2:
        reconstruction_in_channels = int(recon_weight.shape[1])
        reconstruction_use_fft_residual = reconstruction_in_channels >= 10

    npr_scales = artifact_build_meta.get("npr_scales", ckpt.get("npr_scales", None))
    if npr_scales is None and artifact_state:
        npr_weight = artifact_state.get("npr_branch.branch.stem.block.0.weight", None)
        if torch.is_tensor(npr_weight) and npr_weight.ndim >= 2:
            npr_in_channels = int(npr_weight.shape[1])
            n_scales = max(npr_in_channels // 6, 1)
            if n_scales == 1:
                npr_scales = [0.5]
            elif n_scales == 3:
                npr_scales = [0.25, 0.5, 0.75]

    return {
        "checkpoint_type": checkpoint_type,
        "artifact_branches": list(artifact_branches) if artifact_branches is not None else [],
        "reconstruction_in_channels": reconstruction_in_channels,
        "reconstruction_use_fft_residual": bool(reconstruction_use_fft_residual),
        "npr_scales": list(npr_scales) if npr_scales is not None else [],
    }


def _extract_prefixed_state_dict(source: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in source.items():
        if isinstance(key, str) and key.startswith(prefix):
            out[key[len(prefix):]] = value
    return out


def _load_state_if_present(module, state_dict: Optional[Dict[str, torch.Tensor]], strict: bool = False) -> Tuple[List[str], List[str]]:
    if not state_dict:
        return [], []
    missing, unexpected = module.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected)


def load_semantic_branch_checkpoint(model, checkpoint_path: str, device: str, load_projection: bool = False) -> Dict[str, object]:
    if not checkpoint_path:
        return {"loaded": False}
    ckpt = torch.load(checkpoint_path, map_location=device)
    info = {"loaded": True, "checkpoint_path": checkpoint_path, "checkpoint_type": ckpt.get("checkpoint_type", "unknown") if isinstance(ckpt, dict) else "raw"}

    clip_state = {}
    semantic_classifier_state = {}
    semantic_proj_state = {}

    if isinstance(ckpt, dict) and isinstance(ckpt.get("clip_model_state_dict"), dict):
        clip_state = ckpt["clip_model_state_dict"]
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("clip_delta_state_dict"), dict):
        # Compact release format: apply the fine-tuned CLIP delta on top of the
        # base ViT-L/14 model constructed by FrozenBranchAIDEStyleFusionDetector.
        clip_state = ckpt["clip_delta_state_dict"]
    if isinstance(ckpt, dict) and isinstance(ckpt.get("semantic_classifier_state_dict"), dict):
        semantic_classifier_state = ckpt["semantic_classifier_state_dict"]
    if isinstance(ckpt, dict) and isinstance(ckpt.get("semantic_global_proj_state_dict"), dict):
        semantic_proj_state = ckpt["semantic_global_proj_state_dict"]

    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_state_dict"), dict):
        model_state = ckpt["model_state_dict"]
        if not clip_state:
            clip_state = _extract_prefixed_state_dict(model_state, "clip_model.")
        if not semantic_classifier_state:
            semantic_classifier_state = _extract_prefixed_state_dict(model_state, "semantic_classifier.")
        if not semantic_proj_state:
            semantic_proj_state = _extract_prefixed_state_dict(model_state, "semantic_global_proj.")
    elif isinstance(ckpt, dict):
        if not clip_state:
            clip_state = _extract_prefixed_state_dict(ckpt, "clip_model.")
        if not semantic_classifier_state:
            semantic_classifier_state = _extract_prefixed_state_dict(ckpt, "semantic_classifier.")
        if not semantic_proj_state:
            semantic_proj_state = _extract_prefixed_state_dict(ckpt, "semantic_global_proj.")

    miss_clip, unexp_clip = _load_state_if_present(model.clip_model, clip_state, strict=False)
    miss_sem, unexp_sem = _load_state_if_present(model.semantic_classifier, semantic_classifier_state, strict=False)
    info.update({
        "clip_loaded_keys": len(clip_state),
        "semantic_classifier_loaded_keys": len(semantic_classifier_state),
        "clip_missing": miss_clip,
        "clip_unexpected": unexp_clip,
        "semantic_classifier_missing": miss_sem,
        "semantic_classifier_unexpected": unexp_sem,
    })

    if load_projection and semantic_proj_state:
        miss_proj, unexp_proj = _load_state_if_present(model.semantic_global_proj, semantic_proj_state, strict=False)
        info.update({
            "semantic_proj_loaded_keys": len(semantic_proj_state),
            "semantic_proj_missing": miss_proj,
            "semantic_proj_unexpected": unexp_proj,
        })
    else:
        info["semantic_proj_loaded_keys"] = 0

    return info


def load_artifact_branch_checkpoint(model, checkpoint_path: str, device: str, load_projection: bool = False) -> Dict[str, object]:
    if not checkpoint_path:
        return {"loaded": False}
    ckpt = torch.load(checkpoint_path, map_location=device)
    info = {"loaded": True, "checkpoint_path": checkpoint_path, "checkpoint_type": ckpt.get("checkpoint_type", "unknown") if isinstance(ckpt, dict) else "raw"}

    artifact_extractor_state = {}
    artifact_classifier_state = {}
    artifact_proj_state = {}

    if isinstance(ckpt, dict) and isinstance(ckpt.get("artifact_extractor_state_dict"), dict):
        artifact_extractor_state = ckpt["artifact_extractor_state_dict"]
    if isinstance(ckpt, dict) and isinstance(ckpt.get("artifact_classifier_state_dict"), dict):
        artifact_classifier_state = ckpt["artifact_classifier_state_dict"]
    if isinstance(ckpt, dict) and isinstance(ckpt.get("artifact_global_proj_state_dict"), dict):
        artifact_proj_state = ckpt["artifact_global_proj_state_dict"]

    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_state_dict"), dict):
        model_state = ckpt["model_state_dict"]
        if not artifact_extractor_state:
            artifact_extractor_state = _extract_prefixed_state_dict(model_state, "artifact_extractor.")
        if not artifact_classifier_state:
            artifact_classifier_state = _extract_prefixed_state_dict(model_state, "artifact_classifier.")
        if not artifact_proj_state:
            artifact_proj_state = _extract_prefixed_state_dict(model_state, "artifact_global_proj.")
    elif isinstance(ckpt, dict):
        if not artifact_extractor_state:
            artifact_extractor_state = _extract_prefixed_state_dict(ckpt, "artifact_extractor.")
        if not artifact_classifier_state:
            artifact_classifier_state = _extract_prefixed_state_dict(ckpt, "artifact_classifier.")
        if not artifact_proj_state:
            artifact_proj_state = _extract_prefixed_state_dict(ckpt, "artifact_global_proj.")

    miss_ext, unexp_ext = _load_state_if_present(model.artifact_extractor, artifact_extractor_state, strict=False)
    miss_cls, unexp_cls = _load_state_if_present(model.artifact_classifier, artifact_classifier_state, strict=False)
    info.update({
        "artifact_extractor_loaded_keys": len(artifact_extractor_state),
        "artifact_classifier_loaded_keys": len(artifact_classifier_state),
        "artifact_extractor_missing": miss_ext,
        "artifact_extractor_unexpected": unexp_ext,
        "artifact_classifier_missing": miss_cls,
        "artifact_classifier_unexpected": unexp_cls,
    })

    if load_projection and artifact_proj_state:
        miss_proj, unexp_proj = _load_state_if_present(model.artifact_global_proj, artifact_proj_state, strict=False)
        info.update({
            "artifact_proj_loaded_keys": len(artifact_proj_state),
            "artifact_proj_missing": miss_proj,
            "artifact_proj_unexpected": unexp_proj,
        })
    else:
        info["artifact_proj_loaded_keys"] = 0

    return info


def set_requires_grad(module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad


# ---------------------------
# CO-SPY-style output helpers
# ---------------------------

def compute_binary_metrics(y_true, y_pred, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return {"size": 0, "AP": None, "Accuracy": None, "RealAccuracy": None, "FakeAccuracy": None}

    ap = None
    if len(np.unique(y_true)) > 1:
        try:
            from sklearn.metrics import average_precision_score
            ap = float(average_precision_score(y_true, y_pred))
        except Exception:
            ap = None

    pred_bin = (y_pred > threshold).astype(np.int64)
    accuracy = float((pred_bin == y_true).mean())

    real_mask = y_true == 0
    fake_mask = y_true == 1
    real_acc = float((pred_bin[real_mask] == y_true[real_mask]).mean()) if real_mask.any() else None
    fake_acc = float((pred_bin[fake_mask] == y_true[fake_mask]).mean()) if fake_mask.any() else None

    return {
        "size": int(len(y_true)),
        "AP": ap,
        "Accuracy": accuracy,
        "RealAccuracy": real_acc,
        "FakeAccuracy": fake_acc,
    }


def summarize_group_results(group_results: Dict[str, Dict[str, float]]):
    valid_keys = [k for k in group_results.keys() if k not in {"Overall", "Average"}]
    if not valid_keys:
        return group_results

    def _mean(metric_name):
        vals = [group_results[k][metric_name] for k in valid_keys if group_results[k].get(metric_name) is not None]
        return float(np.mean(vals)) if vals else None

    group_results["Average"] = {
        "size": int(len(valid_keys)),
        "AP": _mean("AP"),
        "Accuracy": _mean("Accuracy"),
        "RealAccuracy": _mean("RealAccuracy"),
        "FakeAccuracy": _mean("FakeAccuracy"),
    }
    return group_results


def evaluate_predictions(y_true: List[int], y_pred: List[float], models: List[str], threshold: float = 0.5):
    overall = compute_binary_metrics(y_true, y_pred, threshold=threshold)
    by_model_true = defaultdict(list)
    by_model_pred = defaultdict(list)
    for yt, yp, model in zip(y_true, y_pred, models):
        by_model_true[model].append(int(yt))
        by_model_pred[model].append(float(yp))

    results = {}
    outputs = {}
    for model in sorted(by_model_true.keys()):
        results[model] = compute_binary_metrics(by_model_true[model], by_model_pred[model], threshold=threshold)
        outputs[model] = {
            "y_true": [int(v) for v in by_model_true[model]],
            "y_pred": [float(v) for v in by_model_pred[model]],
        }
    results["Overall"] = overall
    summarize_group_results(results)
    return results, outputs


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_threshold_metric_key(metric: str) -> str:
    metric_map = {
        "balanced_acc": "BalancedAccuracy",
        "accuracy": "Accuracy",
        "fake_acc": "FakeAccuracy",
        "f1": "F1",
        "macro_accuracy": "MacroAccuracy",
        "macro_balanced_acc": "MacroBalancedAccuracy",
        "macro_fake_acc": "MacroFakeAccuracy",
        "macro_f1": "MacroF1",
    }
    if metric not in metric_map:
        raise ValueError(f"Unsupported threshold metric: {metric}")
    return metric_map[metric]


def extract_metric_value(metrics: Dict[str, Optional[float]], metric: str) -> float:
    target_key = get_threshold_metric_key(metric)
    metric_value = metrics.get(target_key, None)
    if metric_value is None or not np.isfinite(metric_value):
        return float('-inf')
    return float(metric_value)


def compute_threshold_metrics(
    y_true: List[int],
    y_pred: List[float],
    threshold: float,
    model_names: Optional[List[str]] = None,
) -> Dict[str, Optional[float]]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.size == 0:
        return {
            "threshold": float(threshold),
            "size": 0,
            "AP": None,
            "Accuracy": None,
            "RealAccuracy": None,
            "FakeAccuracy": None,
            "BalancedAccuracy": None,
            "Precision": None,
            "Recall": None,
            "F1": None,
            "NumModels": 0,
            "MacroAccuracy": None,
            "MacroBalancedAccuracy": None,
            "MacroFakeAccuracy": None,
            "MacroF1": None,
        }

    base_metrics = compute_binary_metrics(y_true_arr, y_pred_arr, threshold=threshold)
    pred_bin = (y_pred_arr > float(threshold)).astype(np.int64)
    tp = int(((pred_bin == 1) & (y_true_arr == 1)).sum())
    pred_pos = int((pred_bin == 1).sum())
    precision = float(tp / pred_pos) if pred_pos > 0 else 0.0
    recall = base_metrics["FakeAccuracy"] if base_metrics["FakeAccuracy"] is not None else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = 0.0 if recall is not None else None
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    real_acc = base_metrics["RealAccuracy"]
    fake_acc = base_metrics["FakeAccuracy"]
    balanced_acc = None
    valid_bacc = [v for v in [real_acc, fake_acc] if v is not None]
    if valid_bacc:
        balanced_acc = float(np.mean(valid_bacc))

    stats = {
        "threshold": float(threshold),
        "size": int(base_metrics["size"]),
        "AP": base_metrics["AP"],
        "Accuracy": base_metrics["Accuracy"],
        "RealAccuracy": real_acc,
        "FakeAccuracy": fake_acc,
        "BalancedAccuracy": balanced_acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "NumModels": 0,
        "MacroAccuracy": None,
        "MacroBalancedAccuracy": None,
        "MacroFakeAccuracy": None,
        "MacroF1": None,
    }

    if model_names is None:
        return stats

    model_names_arr = np.asarray(model_names, dtype=object)
    if model_names_arr.size != y_true_arr.size:
        raise ValueError(
            f"model_names size mismatch: got {model_names_arr.size}, expected {y_true_arr.size}"
        )

    by_model_true = defaultdict(list)
    by_model_pred = defaultdict(list)
    for yt, yp, model_name in zip(y_true_arr.tolist(), y_pred_arr.tolist(), model_names_arr.tolist()):
        by_model_true[str(model_name)].append(int(yt))
        by_model_pred[str(model_name)].append(float(yp))

    per_model_stats = {}
    for model_name in sorted(by_model_true.keys()):
        per_model_stats[model_name] = compute_threshold_metrics(
            by_model_true[model_name],
            by_model_pred[model_name],
            float(threshold),
            model_names=None,
        )

    def _macro_mean(key: str) -> Optional[float]:
        vals = [per_model_stats[m].get(key) for m in per_model_stats if per_model_stats[m].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    stats.update({
        "NumModels": int(len(per_model_stats)),
        "MacroAccuracy": _macro_mean("Accuracy"),
        "MacroBalancedAccuracy": _macro_mean("BalancedAccuracy"),
        "MacroFakeAccuracy": _macro_mean("FakeAccuracy"),
        "MacroF1": _macro_mean("F1"),
    })
    return stats


def search_best_threshold(
    y_true: List[int],
    y_pred: List[float],
    metric: str = "balanced_acc",
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    num_steps: int = 181,
    model_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Optional[float]], List[Dict[str, Optional[float]]]]:
    target_key = get_threshold_metric_key(metric)

    thresholds = np.linspace(float(threshold_min), float(threshold_max), int(num_steps))
    curve_rows: List[Dict[str, Optional[float]]] = []
    best_threshold = 0.5
    best_metrics: Optional[Dict[str, Optional[float]]] = None
    best_value = float('-inf')

    for threshold in thresholds:
        stats = compute_threshold_metrics(y_true, y_pred, float(threshold), model_names=model_names)
        metric_value = stats.get(target_key, None)
        comparable = float(metric_value) if metric_value is not None and np.isfinite(metric_value) else float('-inf')
        curve_rows.append(stats)
        if comparable > best_value:
            best_value = comparable
            best_threshold = float(threshold)
            best_metrics = stats

    if best_metrics is None:
        best_metrics = compute_threshold_metrics(y_true, y_pred, 0.5, model_names=model_names)
        best_threshold = 0.5
    return float(best_threshold), best_metrics, curve_rows


@torch.no_grad()
def collect_head_predictions(model, loader, device: str, use_amp: bool, amp_dtype: str, head: str = "final", desc: str = "Export"):
    model.eval()
    y_true, y_pred, model_names = [], [], []

    iterator = loader
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    except Exception:
        iterator = loader

    head_key_map = {
        "final": "final_output",
        "semantic": "semantic_output",
        "artifact": "artifact_output",
    }
    if head not in head_key_map:
        raise ValueError(f"Unsupported head: {head}")
    logit_key = head_key_map[head]

    for batch in iterator:
        if batch is None:
            continue
        inputs, labels, categories = unpack_fusion_batch(batch, device)

        if use_amp and device.startswith("cuda"):
            dtype = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(inputs, return_details=True)
        else:
            outputs = model(inputs, return_details=True)

        probs = torch.sigmoid(outputs[logit_key].view(-1)).detach().float().cpu().tolist()
        y_pred.extend([float(v) for v in probs])
        y_true.extend([int(v) for v in labels.detach().cpu().tolist()])
        model_names.extend([str(v).replace("\\", "/").split("/")[0] for v in categories])

    return y_true, y_pred, model_names


@torch.no_grad()
def collect_gate_diagnostics(model, loader, device: str, use_amp: bool, amp_dtype: str, split_name: str, desc: str = "Gate Diagnostics"):
    model.eval()
    rows: List[Dict[str, object]] = []

    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    sample_index = 0
    for batch in iterator:
        if batch is None:
            continue
        inputs, labels, categories = unpack_fusion_batch(batch, device)

        if use_amp and device.startswith("cuda"):
            dtype = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(inputs, return_details=True)
        else:
            outputs = model(inputs, return_details=True)

        final_logit = outputs["final_output"].view(-1).detach().float().cpu()
        semantic_logit = outputs["semantic_output"].view(-1).detach().float().cpu()
        artifact_logit = outputs["artifact_output"].view(-1).detach().float().cpu()
        clip_nn_logit_tensor = outputs.get("clip_nn_logit", torch.full_like(outputs["final_output"], float("nan"))).view(-1).detach().float().cpu()
        clip_nn_prob_tensor = outputs.get("clip_nn_prob", torch.full_like(outputs["final_output"], float("nan"))).view(-1).detach().float().cpu()
        fusion_gate = outputs.get("fusion_gate", torch.full_like(outputs["final_output"], float("nan"))).view(-1).detach().float().cpu()
        fusion_bias = outputs.get("fusion_bias", torch.full_like(outputs["final_output"], float("nan"))).view(-1).detach().float().cpu()
        mixed_logit = outputs.get("mixed_logit", torch.full_like(outputs["final_output"], float("nan"))).view(-1).detach().float().cpu()
        mixed_prob_tensor = outputs.get("mixed_prob", torch.full_like(outputs["final_output"], float("nan"))).view(-1).detach().float().cpu()
        semantic_logit_calibrated = outputs.get("semantic_logit_calibrated", torch.full_like(outputs["semantic_output"], float("nan"))).view(-1).detach().float().cpu()
        artifact_logit_calibrated = outputs.get("artifact_logit_calibrated", torch.full_like(outputs["artifact_output"], float("nan"))).view(-1).detach().float().cpu()
        semantic_logit_for_mix = outputs.get("semantic_logit_for_mix", torch.full_like(outputs["semantic_output"], float("nan"))).view(-1).detach().float().cpu()
        artifact_logit_for_mix = outputs.get("artifact_logit_for_mix", torch.full_like(outputs["artifact_output"], float("nan"))).view(-1).detach().float().cpu()
        learned_final_prob_tensor = outputs.get("learned_final_prob", torch.full_like(outputs["final_output"], float("nan"))).view(-1).detach().float().cpu()
        fixed_prob_tensor = outputs.get("fixed_prob", torch.full_like(outputs["final_output"], float("nan"))).view(-1).detach().float().cpu()
        semantic_safe_mask_tensor = outputs.get("semantic_safe_mask", torch.zeros_like(outputs["final_output"])).view(-1).detach().float().cpu()
        semantic_temperature = float(outputs.get("semantic_temperature", torch.tensor([float("nan")], device=labels.device)).view(-1)[0].detach().float().cpu().item())
        artifact_temperature = float(outputs.get("artifact_temperature", torch.tensor([float("nan")], device=labels.device)).view(-1)[0].detach().float().cpu().item())
        semantic_branch_bias = float(outputs.get("semantic_branch_bias", torch.tensor([float("nan")], device=labels.device)).view(-1)[0].detach().float().cpu().item())
        artifact_branch_bias = float(outputs.get("artifact_branch_bias", torch.tensor([float("nan")], device=labels.device)).view(-1)[0].detach().float().cpu().item())

        labels_cpu = labels.detach().float().cpu()
        sem_bce = F.binary_cross_entropy_with_logits(semantic_logit, labels_cpu, reduction="none")
        art_bce = F.binary_cross_entropy_with_logits(artifact_logit, labels_cpu, reduction="none")
        final_bce = F.binary_cross_entropy_with_logits(final_logit, labels_cpu, reduction="none")
        router_target_bce = (art_bce < sem_bce).float()

        final_prob = torch.sigmoid(final_logit)
        semantic_prob = torch.sigmoid(semantic_logit)
        artifact_prob = torch.sigmoid(artifact_logit)

        for i in range(labels_cpu.numel()):
            y = int(labels_cpu[i].item())
            sem_p = float(semantic_prob[i].item())
            art_p = float(artifact_prob[i].item())
            final_p = float(final_prob[i].item())
            sem_pred = int(sem_p > 0.5)
            art_pred = int(art_p > 0.5)
            final_pred = int(final_p > 0.5)
            rows.append({
                "split": split_name,
                "index": sample_index,
                "model": str(categories[i]).replace("\\", "/").split("/")[0],
                "category": str(categories[i]),
                "label": y,
                "semantic_logit": float(semantic_logit[i].item()),
                "artifact_logit": float(artifact_logit[i].item()),
                "clip_nn_logit": float(clip_nn_logit_tensor[i].item()),
                "clip_nn_prob": float(clip_nn_prob_tensor[i].item()),
                "mixed_logit": float(mixed_logit[i].item()),
                "mixed_prob": float(mixed_prob_tensor[i].item()),
                "fusion_bias": float(fusion_bias[i].item()),
                "final_logit": float(final_logit[i].item()),
                "learned_final_prob": float(learned_final_prob_tensor[i].item()),
                "fixed_prob": float(fixed_prob_tensor[i].item()),
                "semantic_safe_mask": int(semantic_safe_mask_tensor[i].item() > 0.5),
                "semantic_logit_calibrated": float(semantic_logit_calibrated[i].item()),
                "artifact_logit_calibrated": float(artifact_logit_calibrated[i].item()),
                "semantic_logit_for_mix": float(semantic_logit_for_mix[i].item()),
                "artifact_logit_for_mix": float(artifact_logit_for_mix[i].item()),
                "semantic_temperature": semantic_temperature,
                "artifact_temperature": artifact_temperature,
                "semantic_branch_bias": semantic_branch_bias,
                "artifact_branch_bias": artifact_branch_bias,
                "semantic_prob": sem_p,
                "artifact_prob": art_p,
                "final_prob": final_p,
                "fusion_gate": float(fusion_gate[i].item()),
                "router_target_artifact": float(router_target_bce[i].item()),
                "bce_oracle_branch": "artifact" if int(router_target_bce[i].item()) == 1 else "semantic",
                "macro_router_target_artifact": float("nan"),
                "macro_oracle_branch": "unknown",
                "semantic_bce": float(sem_bce[i].item()),
                "artifact_bce": float(art_bce[i].item()),
                "final_bce": float(final_bce[i].item()),
                "semantic_pred_fixed05": sem_pred,
                "artifact_pred_fixed05": art_pred,
                "final_pred_fixed05": final_pred,
                "semantic_correct_fixed05": int(sem_pred == y),
                "artifact_correct_fixed05": int(art_pred == y),
                "final_correct_fixed05": int(final_pred == y),
            })
            sample_index += 1

    annotate_macroacc_gate_targets(rows)
    return rows


def annotate_macroacc_gate_targets(rows: List[Dict[str, object]]) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("model", "unknown"))].append(row)
    for _, group_rows in grouped.items():
        sem_acc = float(np.mean([int(r["semantic_correct_fixed05"]) for r in group_rows])) if group_rows else 0.0
        art_acc = float(np.mean([int(r["artifact_correct_fixed05"]) for r in group_rows])) if group_rows else 0.0
        target = 1.0 if art_acc > sem_acc else 0.0
        branch = "artifact" if target == 1.0 else "semantic"
        for r in group_rows:
            r["macro_router_target_artifact"] = target
            r["macro_oracle_branch"] = branch
            r["model_semantic_acc_fixed05"] = sem_acc
            r["model_artifact_acc_fixed05"] = art_acc


def write_gate_diagnostics_csv(rows: List[Dict[str, object]], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_gate_diagnostics(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("model", "unknown"))].append(row)
    summary: List[Dict[str, object]] = []
    for model_name in sorted(grouped.keys()):
        g = grouped[model_name]
        n = max(len(g), 1)
        summary.append({
            "model": model_name,
            "num_samples": len(g),
            "mean_fusion_gate": float(np.mean([float(r["fusion_gate"]) for r in g])),
            "mean_router_target_artifact_bce": float(np.mean([float(r["router_target_artifact"]) for r in g])),
            "macro_router_target_artifact": float(np.mean([float(r.get("macro_router_target_artifact", 0.0)) for r in g])),
            "semantic_acc_fixed05": float(np.mean([int(r["semantic_correct_fixed05"]) for r in g])),
            "artifact_acc_fixed05": float(np.mean([int(r["artifact_correct_fixed05"]) for r in g])),
            "final_acc_fixed05": float(np.mean([int(r["final_correct_fixed05"]) for r in g])),
            "mean_semantic_prob": float(np.mean([float(r["semantic_prob"]) for r in g])),
            "mean_artifact_prob": float(np.mean([float(r["artifact_prob"]) for r in g])),
            "mean_final_prob": float(np.mean([float(r["final_prob"]) for r in g])),
            "mean_final_minus_best_branch_bce": float(np.mean([
                float(r["final_bce"]) - min(float(r["semantic_bce"]), float(r["artifact_bce"])) for r in g
            ])),
        })
    return summary


# ---------------------------
# Frozen-branch detector
# ---------------------------

class FrozenBranchAIDEStyleFusionDetector(AIDEStyleFusionDetector):
    def __init__(self, *args, artifact_feature_dropout: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.artifact_feature_dropout = float(artifact_feature_dropout)
        self.freeze_semantic_branch = False
        self.freeze_artifact_branch = False

    def forward(self, x, return_details: bool = False):
        semantic_x, artifact_x, clip_nn_logit = self._split_branch_inputs(x)
        semantic_x = self._apply_image_patch_shuffle(semantic_x)
        semantic_features, semantic_tokens = self.encode_semantic(semantic_x)
        semantic_tokens = self._maybe_mask_tokens(semantic_tokens)
        if self.training and self.semantic_dropout > 0:
            semantic_features = F.dropout(semantic_features, p=self.semantic_dropout, training=True)

        artifact_features, artifact_tokens, artifact_attention = self.artifact_extractor(artifact_x)
        if self.training and self.artifact_feature_dropout > 0:
            artifact_features = F.dropout(artifact_features, p=self.artifact_feature_dropout, training=True)

        semantic_output = self.semantic_classifier(semantic_features)
        artifact_output = self.artifact_classifier(artifact_features)
        fusion = self._run_joint_fusion(
            semantic_features,
            artifact_features,
            artifact_attention,
            semantic_output,
            artifact_output,
            clip_nn_logit=clip_nn_logit,
        )
        final_output = fusion["final_output"]

        if not return_details:
            return final_output
        return {
            "final_output": final_output,
            "semantic_output": semantic_output,
            "artifact_output": artifact_output,
            "semantic_features": semantic_features,
            "artifact_features": artifact_features,
            "semantic_tokens": semantic_tokens,
            "artifact_tokens": artifact_tokens,
            "artifact_attention": artifact_attention,
            **fusion,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Self-contained logit calibrator from frozen semantic/artifact branches")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test"], help="train: fit fusion and export json; test: load fusion checkpoint and export json")

    parser.add_argument("--train_root", type=str, default="./dataset/train", help="Fusion training root. If your semantic/artifact branches were trained on train, use val here.")
    parser.add_argument("--val_root", type=str, default="./dataset/val", help="Validation root used for model selection and threshold search")
    parser.add_argument("--test_root", type=str, default="./dataset/test", help="Test root used for final export")
    parser.add_argument("--export_split", type=str, default="test", choices=["val", "test"], help="Which split to export to result.json/output.json")

    parser.add_argument("--checkpoint", type=str, default="./checkpoints/aide_frozen_branch_fusion_best.pth", help="Fusion checkpoint path")
    parser.add_argument("--save_dir", type=str, default="./results/aide_frozen_branch_fusion", help="Directory for CO-SPY-style result.json and output.json")
    parser.add_argument("--metrics_json", type=str, default="./results/aide_frozen_branch_fusion/metrics.json", help="Path for detailed metrics json")
    parser.add_argument("--result_json_name", type=str, default="result.json")
    parser.add_argument("--output_json_name", type=str, default="output.json")
    parser.add_argument("--fixed05_result_json_name", type=str, default="fixed05_result.json")

    parser.add_argument("--semantic_checkpoint", type=str, default="", help="semantic_only checkpoint from full_model")
    parser.add_argument("--artifact_checkpoint", type=str, default="", help="artifact_only checkpoint from full_model")
    parser.add_argument("--load_branch_projections_from_checkpoint", action="store_true", help="Try to load semantic_global_proj / artifact_global_proj if they exist in the checkpoints")

    parser.add_argument("--freeze_loaded_branches", action="store_true", help="Freeze both pretrained branches after loading")
    parser.add_argument("--freeze_semantic_branch", action="store_true", help="Freeze semantic branch after loading")
    parser.add_argument("--freeze_artifact_branch", action="store_true", help="Freeze artifact branch after loading")
    parser.add_argument("--artifact_feature_dropout", type=float, default=0.15, help="Feature dropout applied to artifact features during fusion training")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--clip_model_name", type=str, default="ViT-L/14")
    parser.add_argument("--clip_finetune_last_n", type=int, default=0)
    parser.add_argument("--clip_train_layernorm", action="store_true")
    parser.add_argument("--clip_lr", type=float, default=1e-6)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--semantic_dropout", type=float, default=0.10)
    parser.add_argument("--semantic_token_mask_prob", type=float, default=SEMANTIC_ONLY_SEMANTIC_TOKEN_MASK_PROB_DEFAULT)
    parser.add_argument("--patch_shuffle_prob", type=float, default=DEFAULT_PATCH_SHUFFLE_PROB)
    parser.add_argument("--fusion_dim", type=int, default=512)
    parser.add_argument("--artifact_feature_dim", type=int, default=192)
    parser.add_argument("--artifact_token_dim", type=int, default=128)
    parser.add_argument("--token_pool_size", type=int, default=14)
    parser.add_argument("--artifact_branches", nargs="+", default=None, choices=list(ALL_ARTIFACT_BRANCHES))
    parser.add_argument("--npr_scales", nargs="+", type=float, default=None, help="MS-NPR downsample scales. Defaults to checkpoint meta, then 0.25 0.5 0.75.")
    parser.add_argument("--reconstruction_vae_path", type=str, default="")
    parser.add_argument("--reconstruction_vae_subfolder", type=str, default="")
    parser.add_argument("--reconstruction_vae_dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--reconstruction_use_fft_residual", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threshold_metric", type=str, default="macro_accuracy", choices=["balanced_acc", "accuracy", "fake_acc", "f1", "macro_accuracy", "macro_balanced_acc", "macro_fake_acc", "macro_f1"], help="Metric used by threshold search and checkpoint monitoring. Default is macro_accuracy for this semantic-safe fusion variant.")
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_steps", type=int, default=181)
    parser.add_argument("--checkpoint_selection", type=str, default="fixed05", choices=["fixed05", "threshold_search"], help="Metric source for best checkpoint selection. fixed05 avoids selecting checkpoints only because val threshold search overfits.")
    parser.add_argument("--macro_loss_weight", type=float, default=1.0, help="Weight of model-group mean loss. Use 1.0 for MacroAccuracy-oriented fusion.")
    parser.add_argument("--sample_loss_weight", type=float, default=0.0, help="Weight of plain sample mean loss. Use 0.0 for MacroAccuracy-oriented fusion.")
    parser.add_argument("--gate_reg_weight", type=float, default=0.0, help="Regularization forcing mean gate near 0.5. Set 0 for macroacc routing.")
    parser.add_argument("--bias_reg_weight", type=float, default=0.002)
    parser.add_argument("--regret_weight", type=float, default=0.5, help="Penalty when final model-group loss is worse than the macro-selected branch.")
    parser.add_argument("--branch_consistency_weight", type=float, default=0.0, help="Auxiliary BCE aligning final probability with the macro-selected branch probability. Try 0.1, 0.2, 0.3 for MacroAccuracy-oriented fusion.")
    parser.add_argument("--router_supervision_weight", type=float, default=1.0, help="BCE supervision for fusion_gate. In model_macroacc mode, the target is the higher fixed-0.5 accuracy branch within each model group.")
    parser.add_argument("--router_target_mode", type=str, default="model_macroacc", choices=["model_macroacc", "sample_bce"], help="model_macroacc aligns routing with MacroAccuracy; sample_bce is the older per-sample BCE oracle.")
    parser.add_argument("--regret_target_mode", type=str, default="model_macroacc", choices=["model_macroacc", "sample_bce"], help="model_macroacc penalizes group-level regret against the model-level better branch.")
    parser.add_argument("--fusion_mix_space", type=str, default="clipped_logit", choices=["raw_logit", "clipped_logit", "probability"], help="Keep original gate input, but choose how calibrated branch logits are mixed.")
    parser.add_argument("--semantic_logit_clip", type=float, default=5.0, help="Clip semantic calibrated logit before clipped/probability mixing. <=0 disables semantic clipping.")
    parser.add_argument("--artifact_logit_clip", type=float, default=3.0, help="Clip artifact calibrated logit before clipped/probability mixing. <=0 disables artifact clipping.")
    parser.add_argument("--disable_logit_temperature_calibration", action="store_true", help="Disable learnable branch temperature calibration.")
    parser.add_argument("--initial_semantic_temperature", type=float, default=1.0)
    parser.add_argument("--initial_artifact_temperature", type=float, default=1.0)
    parser.add_argument("--disable_branch_logit_bias", action="store_true", help="Disable learnable branch logit bias calibration.")
    parser.add_argument("--output_mix_mode", type=str, default="learned_gate_with_semantic_safety", choices=["learned_gate", "fixed_prob", "learned_gate_with_semantic_safety", "fixed_prob_with_semantic_safety"], help="Final output mode. This variant supports only learned/fixed mix plus optional semantic safety fallback; artifact-safe and dual-safety are intentionally disabled.")
    parser.add_argument("--fixed_artifact_prob_weight", type=float, default=0.20, help="Artifact probability weight used when --output_mix_mode starts with fixed_prob.")
    parser.add_argument("--semantic_safe_semantic_prob_threshold", type=float, default=0.50, help="Semantic safety triggers when semantic fake probability is above this threshold.")
    parser.add_argument("--semantic_safe_artifact_prob_threshold", type=float, default=0.30, help="Semantic safety triggers when artifact fake probability is below this threshold.")
    parser.add_argument("--use_clip_nn_fusion", action="store_true", help="Enable CLIP nearest-neighbor logit as a third score/reliability input to the fusion calibrator.")
    parser.add_argument("--clip_nn_train_csv", type=str, default="", help="CSV for train split with path and CLIP-NN score/logit/prob columns.")
    parser.add_argument("--clip_nn_val_csv", type=str, default="", help="CSV for val split with path and CLIP-NN score/logit/prob columns.")
    parser.add_argument("--clip_nn_test_csv", type=str, default="", help="CSV for test split with path and CLIP-NN score/logit/prob columns.")
    parser.add_argument("--clip_nn_score_column", type=str, default="auto", help="Column name for CLIP-NN score/logit/prob. Use auto to infer.")
    parser.add_argument("--clip_nn_missing_policy", type=str, default="error", choices=["error", "zero"], help="What to do when a sample path is missing from a CLIP-NN CSV.")
    parser.add_argument("--clip_nn_default_logit", type=float, default=0.0, help="Default CLIP-NN logit when --clip_nn_missing_policy zero is used.")
    parser.add_argument("--clip_nn_logit_clip", type=float, default=5.0, help="Clip CLIP-NN logit before it enters fusion features. <=0 disables clipping.")
    parser.add_argument("--enable_artifact_failure_simulator", action="store_true", help="Train-time only simulator that corrupts artifact logits/features/attention so the gate learns to distrust high-confidence artifact failures. No effect during eval/test.")
    parser.add_argument("--artifact_failure_prob", type=float, default=0.35, help="Per-sample probability of artifact failure simulation during fusion training.")
    parser.add_argument("--artifact_failure_bias_min", type=float, default=4.0, help="Minimum absolute logit bias injected into artifact logit during failure simulation.")
    parser.add_argument("--artifact_failure_bias_max", type=float, default=8.0, help="Maximum absolute logit bias injected into artifact logit during failure simulation.")
    parser.add_argument("--artifact_failure_mode", type=str, default="opposite_semantic", choices=["opposite_semantic", "random_sign", "mixed"], help="How to choose simulated artifact logit failure direction.")
    parser.add_argument("--artifact_failure_feature_dropout", type=float, default=0.50, help="Feature dropout applied to artifact features on simulated failure samples.")
    parser.add_argument("--artifact_failure_attention_prob", type=float, default=0.50, help="Probability of perturbing artifact branch attention on simulated failure samples.")
    parser.add_argument("--artifact_failure_attention_strength", type=float, default=0.70, help="Blend strength toward uniform/noisy attention on simulated failure samples.")
    parser.add_argument("--artifact_failure_gate_weight", type=float, default=0.50, help="Auxiliary BCE forcing fusion_gate toward semantic path on simulated artifact failures.")
    parser.add_argument("--artifact_failure_consistency_weight", type=float, default=0.20, help="Auxiliary MSE forcing corrupted-fusion output close to semantic probability on simulated artifact failures.")
    parser.add_argument("--disagreement_safe_weight", type=float, default=0.50, help="Auxiliary BCE forcing gate toward semantic when semantic is correct, artifact is wrong, and their probabilities strongly disagree on training data.")
    parser.add_argument("--disagreement_margin", type=float, default=0.30, help="Minimum |semantic_prob - artifact_prob| for disagreement-safe loss.")
    parser.add_argument("--gate_csv_name", type=str, default="gate_diagnostics.csv")
    parser.add_argument("--gate_summary_csv_name", type=str, default="gate_summary_by_model.csv")
    parser.add_argument("--train_sampler", type=str, default="model_balanced", choices=["shuffle", "model_balanced"])
    parser.add_argument("--verify_images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = resolve_device(args.device)

    artifact_meta = inspect_artifact_checkpoint_build_meta(args.artifact_checkpoint, ALL_ARTIFACT_BRANCHES) if args.artifact_checkpoint else {}
    artifact_branches = args.artifact_branches or artifact_meta.get("artifact_branches") or list(DEFAULT_ARTIFACT_BRANCHES)
    unsupported_branches = sorted(set(artifact_branches) - set(ALL_ARTIFACT_BRANCHES))
    if unsupported_branches:
        raise ValueError(
            f"Artifact checkpoint/args use unsupported branches for useful3 fusion: {unsupported_branches}. "
            f"Expected branches from {list(ALL_ARTIFACT_BRANCHES)}."
        )
    npr_scales = args.npr_scales or artifact_meta.get("npr_scales") or list(DEFAULT_NPR_SCALES)
    npr_scales = list(normalize_npr_scales(npr_scales))
    reconstruction_use_fft_residual = bool(args.reconstruction_use_fft_residual or artifact_meta.get("reconstruction_use_fft_residual", False))

    if "reconstruction_residual" in artifact_branches and not args.reconstruction_vae_path:
        raise ValueError("The artifact checkpoint uses reconstruction_residual, so --reconstruction_vae_path is required at runtime.")

    print(f"[Original fusion + new branches] artifact_branches={artifact_branches}, npr_scales={npr_scales}")

    train_ds = None
    if args.phase == "train":
        if not args.train_root or not os.path.exists(args.train_root):
            raise FileNotFoundError(f"Fusion training root not found: {args.train_root}")
        train_ds = build_train_dataset(args.train_root, verify_images=args.verify_images)

    if not args.val_root or not os.path.exists(args.val_root):
        raise FileNotFoundError(f"Validation root not found: {args.val_root}")
    val_ds = build_eval_dataset(args.val_root, verify_images=args.verify_images)

    test_ds = None
    if args.test_root and os.path.exists(args.test_root):
        test_ds = build_eval_dataset(args.test_root, verify_images=args.verify_images)

    if args.use_clip_nn_fusion:
        if args.phase == "train" and not args.clip_nn_train_csv:
            raise ValueError("--use_clip_nn_fusion requires --clip_nn_train_csv during training")
        if not args.clip_nn_val_csv:
            raise ValueError("--use_clip_nn_fusion requires --clip_nn_val_csv")
        if test_ds is not None and not args.clip_nn_test_csv:
            raise ValueError("--use_clip_nn_fusion requires --clip_nn_test_csv when --test_root is used")
        if train_ds is not None:
            train_scores = load_clip_nn_score_csv(args.clip_nn_train_csv, score_column=args.clip_nn_score_column)
            train_ds = ClipNNScoreDataset(train_ds, train_scores, missing_policy=args.clip_nn_missing_policy, default_score=args.clip_nn_default_logit)
            print(f"[CLIP-NN fusion] train scores loaded: {len(train_scores)} keys from {args.clip_nn_train_csv}")
        val_scores = load_clip_nn_score_csv(args.clip_nn_val_csv, score_column=args.clip_nn_score_column)
        val_ds = ClipNNScoreDataset(val_ds, val_scores, missing_policy=args.clip_nn_missing_policy, default_score=args.clip_nn_default_logit)
        print(f"[CLIP-NN fusion] val scores loaded: {len(val_scores)} keys from {args.clip_nn_val_csv}")
        if test_ds is not None:
            test_scores = load_clip_nn_score_csv(args.clip_nn_test_csv, score_column=args.clip_nn_score_column)
            test_ds = ClipNNScoreDataset(test_ds, test_scores, missing_policy=args.clip_nn_missing_policy, default_score=args.clip_nn_default_logit)
            print(f"[CLIP-NN fusion] test scores loaded: {len(test_scores)} keys from {args.clip_nn_test_csv}")

    model = FrozenBranchAIDEStyleFusionDetector(
        clip_model_name=args.clip_model_name,
        clip_finetune_last_n=args.clip_finetune_last_n,
        clip_train_layernorm=args.clip_train_layernorm,
        semantic_dropout=args.semantic_dropout,
        semantic_token_mask_prob=args.semantic_token_mask_prob,
        patch_shuffle_prob=args.patch_shuffle_prob,
        fusion_dim=args.fusion_dim,
        artifact_feature_dim=args.artifact_feature_dim,
        artifact_token_dim=args.artifact_token_dim,
        token_pool_size=args.token_pool_size,
        artifact_branches=artifact_branches,
        reconstruction_vae_path=(args.reconstruction_vae_path or None),
        reconstruction_vae_subfolder=(args.reconstruction_vae_subfolder or None),
        reconstruction_vae_dtype=args.reconstruction_vae_dtype,
        reconstruction_use_fft_residual=reconstruction_use_fft_residual,
        npr_scales=npr_scales,
        fusion_mix_space=args.fusion_mix_space,
        semantic_logit_clip=args.semantic_logit_clip,
        artifact_logit_clip=args.artifact_logit_clip,
        use_logit_temperature_calibration=not args.disable_logit_temperature_calibration,
        initial_semantic_temperature=args.initial_semantic_temperature,
        initial_artifact_temperature=args.initial_artifact_temperature,
        use_branch_logit_bias=not args.disable_branch_logit_bias,
        output_mix_mode=args.output_mix_mode,
        fixed_artifact_prob_weight=args.fixed_artifact_prob_weight,
        semantic_safe_semantic_prob_threshold=args.semantic_safe_semantic_prob_threshold,
        semantic_safe_artifact_prob_threshold=args.semantic_safe_artifact_prob_threshold,
        enable_artifact_failure_simulator=args.enable_artifact_failure_simulator,
        artifact_failure_prob=args.artifact_failure_prob,
        artifact_failure_bias_min=args.artifact_failure_bias_min,
        artifact_failure_bias_max=args.artifact_failure_bias_max,
        artifact_failure_mode=args.artifact_failure_mode,
        artifact_failure_feature_dropout=args.artifact_failure_feature_dropout,
        artifact_failure_attention_prob=args.artifact_failure_attention_prob,
        artifact_failure_attention_strength=args.artifact_failure_attention_strength,
        model_device=device,
        artifact_feature_dropout=args.artifact_feature_dropout,
        use_clip_nn_fusion=args.use_clip_nn_fusion,
        clip_nn_logit_clip=args.clip_nn_logit_clip,
    ).to(device)

    load_messages = {}
    if args.semantic_checkpoint:
        load_messages["semantic"] = load_semantic_branch_checkpoint(
            model,
            checkpoint_path=args.semantic_checkpoint,
            device=device,
            load_projection=args.load_branch_projections_from_checkpoint,
        )
    if args.artifact_checkpoint:
        load_messages["artifact"] = load_artifact_branch_checkpoint(
            model,
            checkpoint_path=args.artifact_checkpoint,
            device=device,
            load_projection=args.load_branch_projections_from_checkpoint,
        )

    freeze_semantic = bool(args.freeze_loaded_branches or args.freeze_semantic_branch)
    freeze_artifact = bool(args.freeze_loaded_branches or args.freeze_artifact_branch)

    if freeze_semantic:
        if not args.semantic_checkpoint:
            raise ValueError("--freeze_semantic_branch or --freeze_loaded_branches requires --semantic_checkpoint")
        set_requires_grad(model.clip_model, False)
        set_requires_grad(model.semantic_classifier, False)
        model.semantic_dropout = 0.0
        model.semantic_token_mask_prob = 0.0
        model.patch_shuffle_prob = 0.0
        model.freeze_semantic_branch = True

    if freeze_artifact:
        if not args.artifact_checkpoint:
            raise ValueError("--freeze_artifact_branch or --freeze_loaded_branches requires --artifact_checkpoint")
        set_requires_grad(model.artifact_extractor, False)
        set_requires_grad(model.artifact_classifier, False)
        model.artifact_feature_dropout = 0.0
        model.freeze_artifact_branch = True

    class_weights_source = val_ds.labels if train_ds is None else train_ds.labels
    class_weights = compute_class_weights_from_labels(class_weights_source).to(device)
    criterion = LogitCalibratorLoss(
        class_weights=class_weights,
        final_weight=1.0,
        semantic_aux_weight=0.0,
        artifact_aux_weight=0.0,
        macro_loss_weight=args.macro_loss_weight,
        sample_loss_weight=args.sample_loss_weight,
        gate_reg_weight=args.gate_reg_weight,
        bias_reg_weight=args.bias_reg_weight,
        regret_weight=args.regret_weight,
        branch_consistency_weight=args.branch_consistency_weight,
        router_supervision_weight=args.router_supervision_weight,
        router_target_mode=args.router_target_mode,
        regret_target_mode=args.regret_target_mode,
        artifact_failure_gate_weight=(args.artifact_failure_gate_weight if args.enable_artifact_failure_simulator else 0.0),
        artifact_failure_consistency_weight=(args.artifact_failure_consistency_weight if args.enable_artifact_failure_simulator else 0.0),
        disagreement_safe_weight=args.disagreement_safe_weight,
        disagreement_margin=args.disagreement_margin,
    )
    optimizer = build_optimizer(model, clip_lr=args.clip_lr, base_lr=args.base_lr)
    scaler = create_grad_scaler(args.use_amp and device.startswith("cuda") and args.amp_dtype == "fp16")

    persistent = args.num_workers > 0 and os.name != "nt"
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        collate_fn=forgiving_collate,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=persistent,
            collate_fn=forgiving_collate,
        )

    if args.phase == "test":
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Fusion checkpoint not found: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint, device, optimizer=None)
    else:
        train_sampler = None
        train_shuffle = False

        if args.train_sampler == "model_balanced":
            train_sampler = build_model_balanced_sampler(train_ds)
            train_shuffle = False
        else:
            train_sampler = None
            train_shuffle = True

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=train_shuffle if train_sampler is None else False,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=persistent,
            drop_last=True,
            collate_fn=forgiving_collate,
        )
        best_metric = -1.0
        patience_counter = 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                args.use_amp,
                args.amp_dtype,
                scaler,
                args.grad_clip,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            val_metrics = evaluate(model, val_loader, criterion, device, args.use_amp, args.amp_dtype, desc=f"Epoch {epoch}/{args.epochs} - Val")
            epoch_val_y_true, epoch_val_y_pred, epoch_val_model_names = collect_head_predictions(
                model,
                val_loader,
                device=device,
                use_amp=args.use_amp,
                amp_dtype=args.amp_dtype,
                head="final",
                desc=f"Epoch {epoch}/{args.epochs} - Select Threshold",
            )
            epoch_best_threshold, epoch_best_val_metrics, _ = search_best_threshold(
                epoch_val_y_true,
                epoch_val_y_pred,
                metric=args.threshold_metric,
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                num_steps=args.threshold_steps,
                model_names=epoch_val_model_names,
            )
            if args.checkpoint_selection == "fixed05":
                current_metrics = val_metrics["final"]
                current = extract_metric_value(current_metrics, args.threshold_metric)
                monitor_threshold = 0.5
            else:
                current_metrics = epoch_best_val_metrics
                current = extract_metric_value(current_metrics, args.threshold_metric)
                monitor_threshold = epoch_best_threshold
            print(
                f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.5f} | "
                f"val_final_acc@0.5={val_metrics['final']['Accuracy']:.4f} | "
                f"val_final_macro_bacc@0.5={val_metrics['final']['MacroBalancedAccuracy']:.4f} | "
                f"val_sem_acc@0.5={val_metrics['semantic']['Accuracy']:.4f} | "
                f"val_art_acc@0.5={val_metrics['artifact']['Accuracy']:.4f} | "
                f"monitor={args.checkpoint_selection}:{args.threshold_metric}:{current:.4f} | "
                f"monitor_thr={monitor_threshold:.3f} | searched_thr={epoch_best_threshold:.3f}"
            )
            epoch_checkpoint_path = build_epoch_checkpoint_path(args.checkpoint, epoch)
            save_checkpoint(model, optimizer, epoch=epoch, best_metric=max(best_metric, current), path=epoch_checkpoint_path)
            if current > best_metric:
                best_metric = current
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch=epoch, best_metric=best_metric, path=args.checkpoint)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break
        load_checkpoint(model, args.checkpoint, device, optimizer=None)

    val_eval_fixed05 = evaluate(model, val_loader, criterion, device, args.use_amp, args.amp_dtype, desc="Final Val")
    test_eval_fixed05 = None
    if test_loader is not None:
        test_eval_fixed05 = evaluate(model, test_loader, criterion, device, args.use_amp, args.amp_dtype, desc="Final Test")

    val_y_true, val_y_pred, val_model_names = collect_head_predictions(
        model,
        val_loader,
        device=device,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        head="final",
        desc="Collect Val Final Probs",
    )
    best_threshold, best_threshold_val_metrics, threshold_curve = search_best_threshold(
        val_y_true,
        val_y_pred,
        metric=args.threshold_metric,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        num_steps=args.threshold_steps,
        model_names=val_model_names,
    )

    test_predictions_cache = None
    if test_loader is not None:
        test_predictions_cache = collect_head_predictions(
            model,
            test_loader,
            device=device,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
            head="final",
            desc="Collect Test Final Probs",
        )

    metrics_output = {
        "config": {
            "self_contained": True,
            "artifact_branches": artifact_branches,
            "reconstruction_use_fft_residual": reconstruction_use_fft_residual,
            "semantic_checkpoint": args.semantic_checkpoint,
            "artifact_checkpoint": args.artifact_checkpoint,
            "freeze_semantic_branch": freeze_semantic,
            "freeze_artifact_branch": freeze_artifact,
            "threshold_metric": args.threshold_metric,
            "threshold_min": float(args.threshold_min),
            "threshold_max": float(args.threshold_max),
            "threshold_steps": int(args.threshold_steps),
            "checkpoint_selection": args.checkpoint_selection,
            "macro_loss_weight": float(args.macro_loss_weight),
            "sample_loss_weight": float(args.sample_loss_weight),
            "gate_reg_weight": float(args.gate_reg_weight),
            "bias_reg_weight": float(args.bias_reg_weight),
            "regret_weight": float(args.regret_weight),
            "branch_consistency_weight": float(args.branch_consistency_weight),
            "router_supervision_weight": float(args.router_supervision_weight),
            "router_target_mode": args.router_target_mode,
            "regret_target_mode": args.regret_target_mode,
            "fusion_mix_space": args.fusion_mix_space,
            "semantic_logit_clip": float(args.semantic_logit_clip),
            "artifact_logit_clip": float(args.artifact_logit_clip),
            "use_logit_temperature_calibration": bool(not args.disable_logit_temperature_calibration),
            "initial_semantic_temperature": float(args.initial_semantic_temperature),
            "initial_artifact_temperature": float(args.initial_artifact_temperature),
            "use_branch_logit_bias": bool(not args.disable_branch_logit_bias),
            "output_mix_mode": args.output_mix_mode,
            "fixed_artifact_prob_weight": float(args.fixed_artifact_prob_weight),
            "semantic_safe_semantic_prob_threshold": float(args.semantic_safe_semantic_prob_threshold),
            "semantic_safe_artifact_prob_threshold": float(args.semantic_safe_artifact_prob_threshold),
            "use_clip_nn_fusion": bool(args.use_clip_nn_fusion),
            "clip_nn_train_csv": args.clip_nn_train_csv,
            "clip_nn_val_csv": args.clip_nn_val_csv,
            "clip_nn_test_csv": args.clip_nn_test_csv,
            "clip_nn_score_column": args.clip_nn_score_column,
            "clip_nn_missing_policy": args.clip_nn_missing_policy,
            "clip_nn_logit_clip": float(args.clip_nn_logit_clip),
            "enable_artifact_failure_simulator": bool(args.enable_artifact_failure_simulator),
            "artifact_failure_prob": float(args.artifact_failure_prob),
            "artifact_failure_bias_min": float(args.artifact_failure_bias_min),
            "artifact_failure_bias_max": float(args.artifact_failure_bias_max),
            "artifact_failure_mode": args.artifact_failure_mode,
            "artifact_failure_feature_dropout": float(args.artifact_failure_feature_dropout),
            "artifact_failure_attention_prob": float(args.artifact_failure_attention_prob),
            "artifact_failure_attention_strength": float(args.artifact_failure_attention_strength),
            "artifact_failure_gate_weight": float(args.artifact_failure_gate_weight if args.enable_artifact_failure_simulator else 0.0),
            "artifact_failure_consistency_weight": float(args.artifact_failure_consistency_weight if args.enable_artifact_failure_simulator else 0.0),
            "disagreement_safe_weight": float(args.disagreement_safe_weight),
            "disagreement_margin": float(args.disagreement_margin),
            "artifact_safe_enabled": False,
            "dual_safety_enabled": False,
        },
        "checkpoint_load": load_messages,
        "val_fixed05": val_eval_fixed05,
        "threshold_selection": {
            "source_split": "val",
            "target_head": "final",
            "selection_metric": args.threshold_metric,
            "best_threshold": float(best_threshold),
            "best_threshold_val_metrics": best_threshold_val_metrics,
            "fixed05_val_metrics": compute_threshold_metrics(val_y_true, val_y_pred, 0.5, model_names=val_model_names),
            "threshold_curve": threshold_curve,
        },
    }
    if test_eval_fixed05 is not None:
        test_y_true, test_y_pred, test_model_names = test_predictions_cache
        metrics_output["test_fixed05"] = test_eval_fixed05
        metrics_output["test_selected_threshold"] = compute_threshold_metrics(test_y_true, test_y_pred, best_threshold, model_names=test_model_names)

    os.makedirs(args.save_dir, exist_ok=True)

    export_split = args.export_split
    if export_split == "test":
        if test_predictions_cache is None:
            raise ValueError("export_split=test but test dataset is unavailable")
        y_true, y_pred, model_names = test_predictions_cache
    else:
        y_true, y_pred, model_names = val_y_true, val_y_pred, val_model_names

    selected_result_json, output_json = evaluate_predictions(y_true, y_pred, model_names, threshold=best_threshold)
    fixed05_result_json, _ = evaluate_predictions(y_true, y_pred, model_names, threshold=0.5)

    selected_result_path = os.path.join(args.save_dir, args.result_json_name)
    fixed05_result_path = os.path.join(args.save_dir, args.fixed05_result_json_name)
    output_json_path = os.path.join(args.save_dir, args.output_json_name)

    save_json(selected_result_json, selected_result_path)
    save_json(fixed05_result_json, fixed05_result_path)
    save_json(output_json, output_json_path)

    gate_loader = test_loader if export_split == "test" else val_loader
    gate_rows = collect_gate_diagnostics(
        model,
        gate_loader,
        device=device,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        split_name=export_split,
        desc=f"Collect {export_split} Gate Diagnostics",
    )
    gate_summary_rows = summarize_gate_diagnostics(gate_rows)
    gate_csv_path = os.path.join(args.save_dir, args.gate_csv_name)
    gate_summary_csv_path = os.path.join(args.save_dir, args.gate_summary_csv_name)
    write_gate_diagnostics_csv(gate_rows, gate_csv_path)
    write_gate_diagnostics_csv(gate_summary_rows, gate_summary_csv_path)

    metrics_output["export"] = {
        "split": export_split,
        "result_json": selected_result_path,
        "fixed05_result_json": fixed05_result_path,
        "output_json": output_json_path,
        "gate_csv": gate_csv_path,
        "gate_summary_csv": gate_summary_csv_path,
        "selected_threshold": float(best_threshold),
        "selected_threshold_metrics": compute_threshold_metrics(y_true, y_pred, best_threshold, model_names=model_names),
        "fixed05_metrics": compute_threshold_metrics(y_true, y_pred, 0.5, model_names=model_names),
    }

    save_json(metrics_output, args.metrics_json)

    print(json.dumps(metrics_output, ensure_ascii=False, indent=2))
    print(json.dumps({
        "result_json": selected_result_path,
        "fixed05_result_json": fixed05_result_path,
        "output_json": output_json_path,
        "gate_csv": gate_csv_path,
        "gate_summary_csv": gate_summary_csv_path,
        "best_threshold": float(best_threshold),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
