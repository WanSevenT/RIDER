#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone semantic-only trainer extracted from full_model.py.

Design goal:
- Keep the semantic-only path behavior as close as possible to full_model.py.
- No dependency on full_model.py.
- Preserve: full train augmentation, CLIP semantic encoder, patch shuffle,
  semantic token masking, semantic checkpoint format, weighted sampler,
  dynamic hard mining, fixed-0.5 macro_accuracy checkpoint selection,
  and optional val-threshold search for reporting.
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

import clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
FULL_MODEL_SEMANTIC_DROPOUT_DEFAULT = 0.10
FULL_MODEL_SEMANTIC_TOKEN_MASK_PROB_DEFAULT = 0.05
SEMANTIC_ONLY_SEMANTIC_DROPOUT_DEFAULT = 0.0
SEMANTIC_ONLY_SEMANTIC_TOKEN_MASK_PROB_DEFAULT = 0.0
DEFAULT_PATCH_SHUFFLE_PROB = 0.50


def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if device_arg in {"auto", "cuda"}:
        return "cuda:0"
    return device_arg


def get_dataparallel_device_ids(device: str) -> List[int]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return []
    visible_count = torch.cuda.device_count()
    if visible_count <= 1:
        return []
    start_idx = int(device.split(":", 1)[1])
    return list(range(start_idx, visible_count))


def maybe_wrap_dataparallel(model: nn.Module, device: str, multi_gpu: bool = False) -> nn.Module:
    if not multi_gpu or not device.startswith("cuda") or not torch.cuda.is_available():
        return model
    device_ids = get_dataparallel_device_ids(device)
    if len(device_ids) <= 1:
        return model
    if isinstance(model, nn.DataParallel):
        return model
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    model = model.to(f"cuda:{device_ids[0]}")
    return model


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
                f"No valid samples found under: {root_dir}. "
                f"Expected folders like model/0_real, model/1_fake, "
                f"or model/category/0_real, model/category/1_fake."
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
        tail = c.split("_")[-1]
        if tail in {"car", "cat", "chair", "horse"}:
            return tail
        return tail
    return c


def build_stage12_sampler(dataset: ImageForgeryDataset, balance_labels: bool = True):
    labels = np.array(dataset.labels, dtype=np.int64)
    cats = np.array(dataset.categories)
    group_keys = np.array([extract_group_key(c) for c in cats])
    group_counts = Counter(group_keys.tolist())

    weights = np.ones(len(labels), dtype=np.float64)
    if balance_labels:
        label_counts = Counter(labels.tolist())
        label_weight = {k: 1.0 / float(v) for k, v in label_counts.items()}
        weights *= np.array([label_weight[int(y)] for y in labels], dtype=np.float64)
    else:
        label_counts = Counter(labels.tolist())

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    sampler.base_weights = weights.copy()
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


def get_train_transform():
    if HAS_ALBU:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.25),
            A.OneOf([
                A.Affine(scale=(0.85, 1.15), translate_percent=(0.08, 0.08), rotate=(-20, 20), shear=(-8, 8), p=1.0),
                A.Perspective(scale=(0.03, 0.08), p=1.0),
            ], p=0.25),
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

    class _Simple:
        def __call__(self, image):
            img = Image.fromarray(image).resize((224, 224))
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = (arr - np.array(CLIP_IMAGE_MEAN, dtype=np.float32)) / np.array(CLIP_IMAGE_STD, dtype=np.float32)
            return {"image": torch.from_numpy(arr).permute(2, 0, 1)}

    return _Simple()


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
    """
    When CLIP is partially finetuned on CUDA, keep trainable CLIP weights in fp32.
    OpenAI CLIP is often loaded in half precision on GPU; updating those weights
    directly with AdamW can produce non-finite values during finetuning.
    """
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


class SemanticOnlyLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, pos_scale: float = 1.5, focal_gamma: float = 2.0, focal_alpha: float = 0.65):
        super().__init__()
        if class_weights is not None and class_weights.numel() >= 2:
            pos_weight = (class_weights[1] / class_weights[0]).clamp(min=1e-6) * pos_scale
        else:
            pos_weight = torch.tensor([pos_scale], dtype=torch.float32)
        self.loss_fn = WeightedFocalBCE(pos_weight=pos_weight.float(), gamma=focal_gamma, alpha=focal_alpha)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor):
        targets = targets.float()
        semantic_loss = self.loss_fn(outputs["semantic_output"], targets)
        info = {
            "total_loss": float(semantic_loss.item()),
            "final_loss": float("nan"),
            "semantic_loss": float(semantic_loss.item()),
            "artifact_loss": float("nan"),
        }
        return semantic_loss, info


class SemanticOnlyBranchDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        clip_model_name: str = "ViT-L/14",
        clip_finetune_last_n: int = 0,
        clip_train_layernorm: bool = True,
        semantic_dropout: float = SEMANTIC_ONLY_SEMANTIC_DROPOUT_DEFAULT,
        semantic_token_mask_prob: float = SEMANTIC_ONLY_SEMANTIC_TOKEN_MASK_PROB_DEFAULT,
        patch_shuffle_prob: float = DEFAULT_PATCH_SHUFFLE_PROB,
        model_device: str = "cpu",
    ):
        super().__init__()
        self.model_device = model_device
        self.clip_model = load_clip_model(clip_model_name, device=model_device)
        self.feature_dim = self.clip_model.visual.output_dim
        self.semantic_dropout = float(semantic_dropout)
        self.semantic_token_mask_prob = float(semantic_token_mask_prob)
        self.patch_shuffle_prob = float(patch_shuffle_prob)
        self.runtime_train_mode = "semantic_only"

        if clip_finetune_last_n > 0:
            set_clip_finetune(self.clip_model, clip_finetune_last_n, clip_train_layernorm)
            self.clip_model = cast_clip_to_fp32_for_finetune(self.clip_model)
            self.clip_trainable = True
        else:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.clip_trainable = False

        self.semantic_classifier = nn.Linear(self.feature_dim, num_classes)

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
        if torch.rand(1).item() > self.patch_shuffle_prob:
            return x
        b, c, h, w = x.shape
        patch_h, patch_w = h // grid_size, w // grid_size
        x_unfold = x.view(b, c, grid_size, patch_h, grid_size, patch_w)
        patches = x_unfold.permute(0, 2, 4, 1, 3, 5).reshape(b, grid_size * grid_size, c, patch_h, patch_w)
        shuffled_patches = torch.empty_like(patches)
        for i in range(b):
            idx = torch.randperm(grid_size * grid_size, device=x.device)
            shuffled_patches[i] = patches[i, idx]
        x_folded = shuffled_patches.view(b, grid_size, grid_size, c, patch_h, patch_w)
        x_shuffled = x_folded.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return x_shuffled

    def encode_semantic(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.clip_trainable:
            cls_token, patch_tokens = self._encode_clip_tokens(x)
        else:
            with torch.no_grad():
                cls_token, patch_tokens = self._encode_clip_tokens(x)
        return cls_token, patch_tokens

    def forward(self, x, return_details: bool = False):
        semantic_x = x["semantic"] if isinstance(x, dict) else x
        semantic_x_shuffled = self._apply_image_patch_shuffle(semantic_x)
        semantic_features, semantic_tokens = self.encode_semantic(semantic_x_shuffled)
        semantic_tokens = self._maybe_mask_tokens(semantic_tokens)
        if self.training and self.semantic_dropout > 0:
            semantic_features = F.dropout(semantic_features, p=self.semantic_dropout, training=True)
        semantic_output = self.semantic_classifier(semantic_features)
        if not return_details:
            return semantic_output
        return {
            "semantic_output": semantic_output,
            "semantic_features": semantic_features,
            "semantic_tokens": semantic_tokens,
        }


def freeze_for_semantic_only(model: SemanticOnlyBranchDetector, clip_finetune_last_n: int = 0, clip_train_layernorm: bool = True):
    for p in model.parameters():
        p.requires_grad = False
    if clip_finetune_last_n > 0:
        set_clip_finetune(model.clip_model, clip_finetune_last_n, clip_train_layernorm)
        model.clip_model = cast_clip_to_fp32_for_finetune(model.clip_model)
        model.clip_trainable = True
    else:
        for p in model.clip_model.parameters():
            p.requires_grad = False
        model.clip_trainable = False
    for p in model.semantic_classifier.parameters():
        p.requires_grad = True


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def build_optimizer(model: SemanticOnlyBranchDetector, clip_lr: float, base_lr: float = 1e-4):
    clip_params = [p for p in model.clip_model.parameters() if p.requires_grad]
    semantic_params = [p for p in model.semantic_classifier.parameters() if p.requires_grad]
    param_groups = []
    if semantic_params:
        param_groups.append({"params": semantic_params, "lr": base_lr * 0.5})
    if clip_params:
        param_groups.append({"params": clip_params, "lr": clip_lr})
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


def search_best_threshold(df: pd.DataFrame, prob_col: str, metric: str = "macro_balanced_acc", threshold_min: float = 0.05, threshold_max: float = 0.95, num_steps: int = 181) -> Tuple[float, pd.DataFrame]:
    thresholds = np.linspace(threshold_min, threshold_max, num_steps)
    rows = []
    for t in thresholds:
        stats = _compute_threshold_metrics(df, prob_col, float(t))
        stats.update(_compute_group_macro_metrics(df, prob_col, float(t), group_by="model"))
        rows.append({"threshold": float(t), **stats})
    curve_df = pd.DataFrame(rows)
    best_idx = curve_df[metric].idxmax()
    return float(curve_df.loc[best_idx, "threshold"]), curve_df


def move_inputs_to_device(inputs, device: str):
    if torch.is_tensor(inputs):
        return inputs.to(device, non_blocking=True)
    if isinstance(inputs, dict):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    raise TypeError(f"Unsupported input type: {type(inputs)}")


def collect_predictions_single_mode(model: nn.Module, loader: DataLoader, device: str, show_progress: bool = False, progress_desc: str = "predict") -> pd.DataFrame:
    probs, logits, labels, categories, models = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        progress = tqdm(loader, desc=progress_desc, leave=False, dynamic_ncols=True) if show_progress else loader
        for batch in progress:
            if batch is None:
                continue
            x, y, c = batch
            x = move_inputs_to_device(x, device)
            outputs = model(x, return_details=True)
            logit = outputs["semantic_output"].view(-1).detach().float().cpu().numpy()
            prob = torch.sigmoid(outputs["semantic_output"].view(-1)).detach().float().cpu().numpy()
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


def maybe_save_reports(result_dir: str, split_name: str, raw_df: pd.DataFrame, curve_df: pd.DataFrame, report_by_model: pd.DataFrame, report_by_category: pd.DataFrame):
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(os.path.join(result_dir, f"{split_name}_raw_predictions.csv"), index=False)
    curve_df.to_csv(os.path.join(result_dir, f"{split_name}_threshold_curve.csv"), index=False)
    report_by_model.to_csv(os.path.join(result_dir, f"{split_name}_by_model.csv"), index=False)
    report_by_category.to_csv(os.path.join(result_dir, f"{split_name}_by_category.csv"), index=False)


def save_json(obj: Dict[str, object], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_checkpoint_payload(model: SemanticOnlyBranchDetector, optimizer: Optional[optim.Optimizer], scheduler, epoch: int, best_metric: float, args) -> Dict[str, object]:
    base_model = unwrap_model(model)
    return {
        "checkpoint_type": "semantic",
        "epoch": int(epoch),
        "best_val_acc": float(best_metric),
        "monitor_head": "semantic",
        "train_mode": "semantic_only",
        "clip_model_state_dict": base_model.clip_model.state_dict(),
        "semantic_classifier_state_dict": base_model.semantic_classifier.state_dict(),
        "semantic_dropout": float(getattr(base_model, "semantic_dropout", 0.0)),
        "semantic_token_mask_prob": float(getattr(base_model, "semantic_token_mask_prob", 0.0)),
        "patch_shuffle_prob": float(getattr(base_model, "patch_shuffle_prob", 0.0)),
        "clip_trainable": bool(getattr(base_model, "clip_trainable", False)),
        "feature_dim": int(getattr(base_model, "feature_dim", 0)),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "clip_finetune_last_n": int(args.clip_finetune_last_n),
    }


def load_checkpoint(model: SemanticOnlyBranchDetector, checkpoint_path: str, device: str, optimizer: Optional[optim.Optimizer] = None, scheduler = None, resume: bool = False) -> Dict[str, object]:
    if not checkpoint_path:
        return {}
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")
    base_model = unwrap_model(model)
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        if ckpt.get("checkpoint_type", "") == "semantic":
            if "clip_model_state_dict" in ckpt:
                base_model.clip_model.load_state_dict(ckpt["clip_model_state_dict"], strict=False)
            if "semantic_classifier_state_dict" in ckpt:
                base_model.semantic_classifier.load_state_dict(ckpt["semantic_classifier_state_dict"], strict=False)
        else:
            base_model.load_state_dict(ckpt, strict=False)
    if resume and optimizer is not None and isinstance(ckpt.get("optimizer_state_dict"), dict):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass
    if resume and scheduler is not None and isinstance(ckpt.get("scheduler_state_dict"), dict):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            pass
    if getattr(base_model, "clip_trainable", False):
        base_model.clip_model = cast_clip_to_fp32_for_finetune(base_model.clip_model)
    return ckpt


def save_semantic_alignment_sidecar(checkpoint_path: str, model: SemanticOnlyBranchDetector, args) -> str:
    base_model = unwrap_model(model)
    sidecar_path = str(Path(checkpoint_path).with_suffix(".semantic_alignment.json"))
    payload = {
        "semantic_checkpoint": str(checkpoint_path),
        "checkpoint_type": "semantic",
        "clip_model_name": str(args.clip_model_name),
        "clip_finetune_last_n": int(args.clip_finetune_last_n),
        "clip_train_layernorm": bool(args.clip_train_layernorm),
        "semantic_dropout": float(getattr(base_model, "semantic_dropout", 0.0)),
        "semantic_token_mask_prob": float(getattr(base_model, "semantic_token_mask_prob", 0.0)),
        "patch_shuffle_prob": float(getattr(base_model, "patch_shuffle_prob", 0.0)),
        "feature_dim": int(getattr(base_model, "feature_dim", 0)),
    }
    save_json(payload, sidecar_path)
    return sidecar_path


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion, device: str, use_amp: bool, amp_dtype: str, grad_clip: float, show_progress: bool = False, epoch_desc: str = "train") -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    cat_stats = defaultdict(lambda: {"count": 0, "err": 0, "fake_count": 0, "fake_err": 0})
    scaler = create_grad_scaler(enabled=(use_amp and device.startswith("cuda") and amp_dtype.lower() == "fp16"))

    progress = tqdm(loader, desc=epoch_desc, leave=False, dynamic_ncols=True) if show_progress else loader
    for batch in progress:
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
        else:
            outputs = model(inputs, return_details=True)
            loss, _ = criterion(outputs, labels)

        if not torch.isfinite(outputs["semantic_output"]).all():
            raise RuntimeError("Non-finite semantic_output detected during training. When finetuning CLIP, keep CLIP weights in fp32.")
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected during training.")

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
        logits = outputs["semantic_output"].detach().float().view(-1)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += int((preds == labels.long().view(-1)).sum().item())
        total += int(labels.numel())
        for cat, y, pred in zip(categories, labels.long().view(-1).cpu().tolist(), preds.cpu().tolist()):
            st = cat_stats[str(cat).lower()]
            st["count"] += 1
            st["err"] += int(y != pred)
            if int(y) == 1:
                st["fake_count"] += 1
                st["fake_err"] += int(y != pred)

        if show_progress and hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{total_loss / max(total,1):.4f}", acc=f"{100.0 * correct / max(total,1):.2f}%", semantic_logit=f"{outputs['semantic_output'].detach().float().view(-1).mean().item():.4f}")

    epoch_err_stats = {}
    for cat, st in cat_stats.items():
        err_rate = st["err"] / max(st["count"], 1)
        fake_err_rate = st["fake_err"] / max(st["fake_count"], 1) if st["fake_count"] > 0 else err_rate
        epoch_err_stats[cat] = {"err_rate": err_rate, "fake_err_rate": fake_err_rate}

    return total_loss / max(total, 1), 100.0 * correct / max(total, 1), epoch_err_stats


@torch.no_grad()
def compute_val_loss(model: nn.Module, loader: DataLoader, criterion, device: str, use_amp: bool, amp_dtype: str, show_progress: bool = False, epoch_desc: str = "val") -> float:
    model.eval()
    loss_sum = 0.0
    total = 0
    progress = tqdm(loader, desc=epoch_desc, leave=False, dynamic_ncols=True) if show_progress else loader
    for batch in progress:
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
        if not torch.isfinite(outputs["semantic_output"]).all():
            raise RuntimeError("Non-finite semantic_output detected during validation.")
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected during validation.")
        loss_sum += float(loss.item()) * labels.numel()
        total += int(labels.numel())
    return loss_sum / max(total, 1)


def evaluate_reports(raw_df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, pd.DataFrame]:
    return {
        "by_model": _evaluate_one_head_from_df(raw_df, "prob", threshold, "model"),
        "by_category": _evaluate_one_head_from_df(raw_df, "prob", threshold, "category"),
    }


def save_and_test_periodic_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler,
    epoch: int,
    monitor_value: float,
    args,
    device: str,
    persistent: bool,
    result_dir: str,
    val_raw: pd.DataFrame,
    test_ds: Optional[ImageForgeryDataset],
) -> Dict[str, object]:
    periodic_dir = str(args.periodic_checkpoint_dir).strip() or os.path.join(result_dir, "periodic")
    Path(periodic_dir).mkdir(parents=True, exist_ok=True)

    epoch_tag = f"epoch_{int(epoch):04d}"
    checkpoint_path = os.path.join(periodic_dir, f"semantic_only_{epoch_tag}.pth")
    torch.save(build_checkpoint_payload(unwrap_model(model), optimizer, scheduler, int(epoch), float(monitor_value), args), checkpoint_path)
    sidecar_path = save_semantic_alignment_sidecar(checkpoint_path, unwrap_model(model), args)

    val_best_thr, val_curve = search_best_threshold(
        val_raw,
        "prob",
        metric=str(args.threshold_metric),
        threshold_min=float(args.threshold_min),
        threshold_max=float(args.threshold_max),
        num_steps=int(args.threshold_steps),
    )
    val_fixed_reports = evaluate_reports(val_raw, threshold=0.5)

    summary: Dict[str, object] = {
        "epoch": int(epoch),
        "checkpoint": str(checkpoint_path),
        "semantic_alignment_sidecar": str(sidecar_path),
        "val_threshold": float(val_best_thr),
        "val_search_metrics": _compute_monitor_metric_from_raw_df(val_raw, "prob", threshold=float(val_best_thr)),
        "val_fixed05_metrics": _compute_monitor_metric_from_raw_df(val_raw, "prob", threshold=0.5),
    }

    if bool(args.save_test_reports):
        maybe_save_reports(
            periodic_dir,
            f"{epoch_tag}_val_fixed05",
            val_raw,
            val_curve,
            val_fixed_reports["by_model"],
            val_fixed_reports["by_category"],
        )

    if test_ds is None:
        summary["test_status"] = "skipped_no_test_root"
        summary_path = os.path.join(periodic_dir, f"{epoch_tag}_summary.json")
        save_json(summary, summary_path)
        print(f"  ✓ Saved periodic checkpoint: {checkpoint_path}")
        print("  ⚠ Skipped periodic test: test_root not found or empty")
        return summary

    eval_batch_size = int(args.batch_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        collate_fn=forgiving_collate,
    )
    test_raw = collect_predictions_single_mode(
        model,
        test_loader,
        device,
        show_progress=bool(args.show_progress),
        progress_desc=f"{epoch_tag}-test-predict",
    )
    test_best_thr, test_curve = search_best_threshold(
        test_raw,
        "prob",
        metric=str(args.threshold_metric),
        threshold_min=float(args.threshold_min),
        threshold_max=float(args.threshold_max),
        num_steps=int(args.threshold_steps),
    )
    test_fixed_reports = evaluate_reports(test_raw, threshold=0.5)
    test_valselected_reports = evaluate_reports(test_raw, threshold=float(val_best_thr))
    test_search_reports = evaluate_reports(test_raw, threshold=float(test_best_thr))

    test_fixed05_metrics = _compute_monitor_metric_from_raw_df(test_raw, "prob", threshold=0.5)
    test_valselected_metrics = _compute_monitor_metric_from_raw_df(test_raw, "prob", threshold=float(val_best_thr))
    test_search_metrics = _compute_monitor_metric_from_raw_df(test_raw, "prob", threshold=float(test_best_thr))
    summary.update({
        "test_threshold": float(test_best_thr),
        "test_search_metrics": test_search_metrics,
        "test_valselected_metrics": test_valselected_metrics,
        "test_fixed05_metrics": test_fixed05_metrics,
    })

    if bool(args.save_test_reports):
        maybe_save_reports(periodic_dir, f"{epoch_tag}_test_fixed05", test_raw, test_curve, test_fixed_reports["by_model"], test_fixed_reports["by_category"])
        maybe_save_reports(periodic_dir, f"{epoch_tag}_test_valselected", test_raw, test_curve, test_valselected_reports["by_model"], test_valselected_reports["by_category"])
        maybe_save_reports(periodic_dir, f"{epoch_tag}_test_search", test_raw, test_curve, test_search_reports["by_model"], test_search_reports["by_category"])

    summary_path = os.path.join(periodic_dir, f"{epoch_tag}_summary.json")
    save_json(summary, summary_path)
    print(f"  ✓ Saved periodic checkpoint: {checkpoint_path}")
    print(
        f"  ✓ Periodic test @ {epoch_tag} | "
        f"fixed05 overallAcc {100.0 * float(test_fixed05_metrics['accuracy']):.2f}% | "
        f"macroAcc {100.0 * float(test_fixed05_metrics['macro_accuracy']):.2f}% | "
        f"macroBAcc {100.0 * float(test_fixed05_metrics['macro_balanced_acc']):.2f}% | "
        f"testBestThr {float(test_best_thr):.4f}"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone semantic-only trainer extracted from full_model.py")
    parser.add_argument("--train_root", type=str, default="./dataset/train")
    parser.add_argument("--val_root", type=str, default="./dataset/val")
    parser.add_argument("--test_root", type=str, default="./dataset/test")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/semantic_only_extracted_best.pth")
    parser.add_argument("--init_semantic_checkpoint", type=str, default="")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_scheduler", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--clip_model_name", type=str, default="ViT-L/14")
    parser.add_argument("--clip_finetune_last_n", type=int, default=0)
    parser.add_argument("--clip_lr", type=float, default=1e-6)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--semantic_dropout", type=float, default=SEMANTIC_ONLY_SEMANTIC_DROPOUT_DEFAULT)
    parser.add_argument("--semantic_token_mask_prob", type=float, default=SEMANTIC_ONLY_SEMANTIC_TOKEN_MASK_PROB_DEFAULT)
    parser.add_argument("--patch_shuffle_prob", type=float, default=DEFAULT_PATCH_SHUFFLE_PROB)
    parser.add_argument("--hard_mining_dynamic_boost", type=float, default=2.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--threshold_metric", type=str, default="macro_balanced_acc", choices=["macro_balanced_acc", "macro_accuracy", "macro_fake_acc", "macro_f1", "balanced_acc", "accuracy", "fake_acc", "f1"])
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_steps", type=int, default=181)
    parser.add_argument("--verify_images", action="store_true")
    parser.add_argument("--save_test_reports", action="store_true")
    parser.add_argument("--save_test_interval", type=int, default=5, help="Save a periodic checkpoint and run test every N epochs. Set <=0 to disable.")
    parser.add_argument("--periodic_checkpoint_dir", type=str, default="", help="Directory for periodic checkpoints and per-interval test summaries. Defaults to <checkpoint_stem>/periodic.")
    parser.add_argument("--show_progress", action="store_true")
    parser.set_defaults(clip_train_layernorm=True)
    clip_ln_group = parser.add_mutually_exclusive_group()
    clip_ln_group.add_argument("--clip_train_layernorm", dest="clip_train_layernorm", action="store_true")
    clip_ln_group.add_argument("--no_clip_train_layernorm", dest="clip_train_layernorm", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(int(args.seed))
    device = resolve_device(args.device)
    persistent = int(args.num_workers) > 0 and os.name != "nt"

    train_ds = None
    if not args.test_only:
        train_ds = ImageForgeryDataset(args.train_root, transform=get_train_transform(), verify_images=args.verify_images)
    val_ds = ImageForgeryDataset(args.val_root, transform=get_val_transform(), verify_images=args.verify_images)
    test_ds = ImageForgeryDataset(args.test_root, transform=get_val_transform(), verify_images=args.verify_images) if args.test_root and os.path.exists(args.test_root) else None

    model = SemanticOnlyBranchDetector(
        clip_model_name=args.clip_model_name,
        clip_finetune_last_n=int(args.clip_finetune_last_n),
        clip_train_layernorm=bool(args.clip_train_layernorm),
        semantic_dropout=float(args.semantic_dropout),
        semantic_token_mask_prob=float(args.semantic_token_mask_prob),
        patch_shuffle_prob=float(args.patch_shuffle_prob),
        model_device=device,
    ).to(device)
    freeze_for_semantic_only(model, clip_finetune_last_n=int(args.clip_finetune_last_n), clip_train_layernorm=bool(args.clip_train_layernorm))
    model = maybe_wrap_dataparallel(model, device, multi_gpu=bool(args.multi_gpu))

    weight_source_labels = train_ds.labels if train_ds is not None else val_ds.labels
    criterion = SemanticOnlyLoss(class_weights=torch.tensor([1.0,1.0], device=device))
    criterion = SemanticOnlyLoss(class_weights=(torch.bincount(torch.tensor(weight_source_labels, dtype=torch.long), minlength=2).float().reciprocal().to(device)), pos_scale=1.5, focal_gamma=2.0, focal_alpha=0.65)

    optimizer = scheduler = None
    if not args.test_only:
        optimizer = build_optimizer(unwrap_model(model), clip_lr=float(args.clip_lr), base_lr=float(args.base_lr))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=2, eta_min=1e-7)

    if args.init_semantic_checkpoint and not args.resume:
        load_checkpoint(model, args.init_semantic_checkpoint, device, optimizer=None, scheduler=None, resume=False)

    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        ckpt = load_checkpoint(model, args.checkpoint, device, optimizer=optimizer, scheduler=scheduler if args.resume_scheduler else None, resume=True)
        start_epoch = int(ckpt.get("epoch", 0) or 0)
        best_metric = float(ckpt.get("best_val_acc", 0.0) or 0.0)

    result_dir = str(Path(args.checkpoint).with_suffix(""))
    sidecar_path = str(Path(args.checkpoint).with_suffix(".semantic_alignment.json"))
    periodic_test_summaries: List[Dict[str, object]] = []

    if not args.test_only:
        train_sampler, _, group_counts, label_counts = build_stage12_sampler(train_ds, balance_labels=True)
        print("✅ Stage12 sampler built")
        print(f"  group_counts(top10): {sorted(group_counts.items(), key=lambda x: -x[1])[:10]}")
        print(f"  label_counts: {label_counts}")
        train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), sampler=train_sampler, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(), persistent_workers=persistent, drop_last=True, collate_fn=forgiving_collate)
        val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(), persistent_workers=persistent, collate_fn=forgiving_collate)

        patience_counter = 0
        for epoch in range(start_epoch, int(args.epochs)):
            train_loss, train_acc, epoch_err_stats = train_one_epoch(model, train_loader, optimizer, criterion, device, bool(args.use_amp), str(args.amp_dtype), float(args.grad_clip), show_progress=bool(args.show_progress), epoch_desc=f"Epoch {epoch+1}/{int(args.epochs)} - Train")
            val_loss = compute_val_loss(model, val_loader, criterion, device, bool(args.use_amp), str(args.amp_dtype), show_progress=bool(args.show_progress), epoch_desc=f"Epoch {epoch+1}/{int(args.epochs)} - Val")

            val_single_df = collect_predictions_single_mode(model, val_loader, device, show_progress=False)
            val_metrics = _compute_monitor_metric_from_raw_df(val_single_df, "prob", threshold=0.5)
            val_acc = 100.0 * float(val_metrics["accuracy"])
            val_macro_acc = 100.0 * float(val_metrics["macro_accuracy"])
            val_macro_balanced = 100.0 * float(val_metrics["macro_balanced_acc"])
            val_macro_fake = 100.0 * float(val_metrics["macro_fake_acc"])
            monitor_value = val_macro_acc
            if scheduler is not None:
                try:
                    scheduler.step()
                except TypeError:
                    scheduler.step(monitor_value)

            print(
                f"Epoch {epoch+1}/{int(args.epochs)} | TrainLoss {train_loss:.5f} semanticAcc {train_acc:.2f}% | "
                f"ValLoss {val_loss:.5f} overallAcc {val_acc:.2f}% | macroAcc {val_macro_acc:.2f}% | "
                f"macroBAcc {val_macro_balanced:.2f}% | macroFake {val_macro_fake:.2f}%"
            )

            if monitor_value > best_metric:
                best_metric = monitor_value
                patience_counter = 0
                torch.save(build_checkpoint_payload(unwrap_model(model), optimizer, scheduler, epoch + 1, best_metric, args), args.checkpoint)
                sidecar_path = save_semantic_alignment_sidecar(args.checkpoint, unwrap_model(model), args)
                print(f"  ✓ Saved best checkpoint (semantic): {args.checkpoint} (Best macro mean Acc: {best_metric:.2f}%)")
            else:
                patience_counter += 1

            interval = int(args.save_test_interval)
            if interval > 0 and ((epoch + 1) % interval == 0):
                periodic_summary = save_and_test_periodic_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    monitor_value,
                    args,
                    device,
                    persistent,
                    result_dir,
                    val_single_df,
                    test_ds,
                )
                periodic_test_summaries.append(periodic_summary)

            if patience_counter >= int(args.patience):
                print(f"  ⏹ Early stopping at epoch {epoch+1} (best macro mean Acc: {best_metric:.2f}%)")
                break

            if isinstance(train_loader.sampler, WeightedRandomSampler):
                update_sampler_for_hard_mining(train_loader.dataset, train_loader.sampler, epoch_err_stats, dynamic_boost=float(args.hard_mining_dynamic_boost))
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    if os.path.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, device, optimizer=None, scheduler=None, resume=False)

    model.eval()
    eval_batch_size = int(args.batch_size)
    val_loader_eval = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(), persistent_workers=persistent, collate_fn=forgiving_collate)
    val_raw = collect_predictions_single_mode(model, val_loader_eval, device, show_progress=bool(args.show_progress), progress_desc="val-predict")
    val_best_thr, val_curve = search_best_threshold(val_raw, "prob", metric=str(args.threshold_metric), threshold_min=float(args.threshold_min), threshold_max=float(args.threshold_max), num_steps=int(args.threshold_steps))
    val_fixed_reports = evaluate_reports(val_raw, threshold=0.5)
    val_valselected_reports = evaluate_reports(val_raw, threshold=float(val_best_thr))

    summary = {
        "checkpoint": str(args.checkpoint),
        "semantic_alignment_sidecar": str(sidecar_path),
        "val_threshold": float(val_best_thr),
        "val_search_metrics": _compute_monitor_metric_from_raw_df(val_raw, "prob", threshold=float(val_best_thr)),
        "val_fixed05_metrics": _compute_monitor_metric_from_raw_df(val_raw, "prob", threshold=0.5),
    }
    if periodic_test_summaries:
        summary["periodic_test_summaries"] = periodic_test_summaries

    if args.save_test_reports:
        maybe_save_reports(result_dir, "val_fixed05", val_raw, val_curve, val_fixed_reports["by_model"], val_fixed_reports["by_category"])

    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(), persistent_workers=persistent, collate_fn=forgiving_collate)
        test_raw = collect_predictions_single_mode(model, test_loader, device, show_progress=bool(args.show_progress), progress_desc="test-predict")
        test_best_thr, test_curve = search_best_threshold(test_raw, "prob", metric=str(args.threshold_metric), threshold_min=float(args.threshold_min), threshold_max=float(args.threshold_max), num_steps=int(args.threshold_steps))
        test_fixed_reports = evaluate_reports(test_raw, threshold=0.5)
        test_valselected_reports = evaluate_reports(test_raw, threshold=float(val_best_thr))
        test_search_reports = evaluate_reports(test_raw, threshold=float(test_best_thr))
        summary.update({
            "test_threshold": float(test_best_thr),
            "test_search_metrics": _compute_monitor_metric_from_raw_df(test_raw, "prob", threshold=float(test_best_thr)),
            "test_valselected_metrics": _compute_monitor_metric_from_raw_df(test_raw, "prob", threshold=float(val_best_thr)),
            "test_fixed05_metrics": _compute_monitor_metric_from_raw_df(test_raw, "prob", threshold=0.5),
        })
        if args.save_test_reports:
            maybe_save_reports(result_dir, "test_fixed05", test_raw, test_curve, test_fixed_reports["by_model"], test_fixed_reports["by_category"])
            maybe_save_reports(result_dir, "test_valselected", test_raw, test_curve, test_valselected_reports["by_model"], test_valselected_reports["by_category"])
            maybe_save_reports(result_dir, "test_search", test_raw, test_curve, test_search_reports["by_model"], test_search_reports["by_category"])

    save_json(summary, os.path.join(result_dir, "semantic_only_summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
