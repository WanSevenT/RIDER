#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

import clip

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if device in {"auto", "cuda"}:
        return "cuda:0"
    return device


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def split_category(category: str) -> Tuple[str, str]:
    parts = [p for p in str(category).replace("\\", "/").strip("/").split("/") if p]
    if not parts:
        return "unknown", ""
    return parts[0], "/".join(parts[1:]) if len(parts) > 1 else ""


def scan_dataset(root_dir: str) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for current_root, dirnames, _ in os.walk(root_dir, topdown=True):
        dirnames.sort()
        has_real = "0_real" in dirnames
        has_fake = "1_fake" in dirnames
        if not (has_real or has_fake):
            continue
        rel_dir = os.path.relpath(current_root, root_dir)
        rel_dir = "" if rel_dir == "." else rel_dir.replace("\\", "/")
        category = rel_dir if rel_dir else Path(current_root).name
        model, subset = split_category(category)
        for dirname, label in [("0_real", 0), ("1_fake", 1)]:
            label_dir = os.path.join(current_root, dirname)
            if not os.path.isdir(label_dir):
                continue
            for img_name in sorted(os.listdir(label_dir)):
                if not img_name.lower().endswith(IMG_EXTS):
                    continue
                path = os.path.join(label_dir, img_name)
                samples.append({
                    "path": path,
                    "label": int(label),
                    "category": category,
                    "model": model,
                    "subset": subset,
                })
        dirnames[:] = [d for d in dirnames if d not in {"0_real", "1_fake"}]
    if not samples:
        raise RuntimeError(f"No images found under {root_dir}")
    return samples


class CLIPFeatureDataset(Dataset):
    def __init__(self, samples: List[Dict[str, object]], preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        try:
            img = Image.open(str(s["path"])).convert("RGB")
            x = self.preprocess(img)
            return x, int(s["label"]), str(s["model"]), str(s["category"]), str(s["path"])
        except Exception:
            return None


def forgiving_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


@torch.no_grad()
def extract_features(samples, preprocess, model, device: str, batch_size: int, num_workers: int) -> Dict[str, object]:
    ds = CLIPFeatureDataset(samples, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=forgiving_collate)
    feats, labels, models, categories, paths = [], [], [], [], []
    model.eval()
    for batch in dl:
        if batch is None:
            continue
        x, y, m, c, p = batch
        x = x.to(device, non_blocking=True)
        f = model.encode_image(x).float()
        f = torch.nn.functional.normalize(f, dim=1)
        feats.append(f.cpu())
        labels.extend([int(v) for v in y.tolist()])
        models.extend([str(v) for v in m])
        categories.extend([str(v) for v in c])
        paths.extend([str(v) for v in p])
    if not feats:
        raise RuntimeError("No features extracted")
    return {
        "features": torch.cat(feats, dim=0),
        "labels": np.asarray(labels, dtype=np.int64),
        "models": np.asarray(models, dtype=object),
        "categories": np.asarray(categories, dtype=object),
        "paths": np.asarray(paths, dtype=object),
    }


def build_balanced_bank(train_pack: Dict[str, object], max_per_model_class: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = train_pack["labels"]
    models = train_pack["models"]
    feats = train_pack["features"]
    paths = train_pack["paths"]
    selected_real, selected_fake = [], []
    for model_name in sorted(set(models.tolist())):
        for label, bucket in [(0, selected_real), (1, selected_fake)]:
            idxs = np.where((models == model_name) & (labels == label))[0]
            if idxs.size == 0:
                continue
            if idxs.size > max_per_model_class:
                idxs = rng.choice(idxs, size=max_per_model_class, replace=False)
            bucket.extend([int(i) for i in idxs.tolist()])
    real_idx = np.asarray(selected_real, dtype=np.int64)
    fake_idx = np.asarray(selected_fake, dtype=np.int64)
    if real_idx.size == 0 or fake_idx.size == 0:
        raise RuntimeError("Bank must contain both real and fake features")
    return {
        "real_features": feats[real_idx],
        "fake_features": feats[fake_idx],
        "real_paths": paths[real_idx],
        "fake_paths": paths[fake_idx],
    }


@torch.no_grad()
def compute_scores(pack: Dict[str, object], bank: Dict[str, object], k: int, device: str, batch_size: int, exclude_self: bool = False):
    feats = pack["features"]
    paths = pack["paths"]
    real_bank = bank["real_features"].to(device)
    fake_bank = bank["fake_features"].to(device)
    real_paths = np.asarray(bank["real_paths"], dtype=object)
    fake_paths = np.asarray(bank["fake_paths"], dtype=object)
    out_sim_real, out_sim_fake, out_score = [], [], []
    k_real = min(int(k), real_bank.shape[0])
    k_fake = min(int(k), fake_bank.shape[0])
    for start in range(0, feats.shape[0], batch_size):
        end = min(start + batch_size, feats.shape[0])
        f = feats[start:end].to(device)
        sim_real = f @ real_bank.t()
        sim_fake = f @ fake_bank.t()
        if exclude_self:
            batch_paths = paths[start:end]
            for local_i, path in enumerate(batch_paths):
                rp = np.where(real_paths == path)[0]
                fp = np.where(fake_paths == path)[0]
                if rp.size > 0:
                    sim_real[local_i, torch.as_tensor(rp, device=device)] = -1e9
                if fp.size > 0:
                    sim_fake[local_i, torch.as_tensor(fp, device=device)] = -1e9
        top_real = sim_real.topk(k=k_real, dim=1).values.mean(dim=1)
        top_fake = sim_fake.topk(k=k_fake, dim=1).values.mean(dim=1)
        score = top_fake - top_real
        out_sim_real.append(top_real.cpu())
        out_sim_fake.append(top_fake.cpu())
        out_score.append(score.cpu())
    return torch.cat(out_sim_real).numpy(), torch.cat(out_sim_fake).numpy(), torch.cat(out_score).numpy()


def write_scores_csv(path: str, pack: Dict[str, object], sim_real, sim_fake, score, tau: float, bias: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "model", "category", "sim_real", "sim_fake", "clip_nn_score", "clip_nn_logit", "clip_nn_prob"])
        writer.writeheader()
        for i in range(len(pack["paths"])):
            logit = float(tau) * float(score[i]) + float(bias)
            writer.writerow({
                "path": str(pack["paths"][i]),
                "label": int(pack["labels"][i]),
                "model": str(pack["models"][i]),
                "category": str(pack["categories"][i]),
                "sim_real": float(sim_real[i]),
                "sim_fake": float(sim_fake[i]),
                "clip_nn_score": float(score[i]),
                "clip_nn_logit": float(logit),
                "clip_nn_prob": float(sigmoid(logit)),
            })


def parse_args():
    p = argparse.ArgumentParser(description="Generate CLIP nearest-neighbor scores for semantic/artifact fusion")
    p.add_argument("--train_root", required=True)
    p.add_argument("--val_root", required=True)
    p.add_argument("--test_root", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--clip_model_name", default="ViT-L/14")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max_per_model_class", type=int, default=500)
    p.add_argument("--tau", type=float, default=40.0)
    p.add_argument("--bias", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = resolve_device(args.device)
    local_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")
    os.makedirs(local_model_dir, exist_ok=True)
    model, preprocess = clip.load(args.clip_model_name, device=device, download_root=local_model_dir)
    model.eval()

    train_samples = scan_dataset(args.train_root)
    val_samples = scan_dataset(args.val_root)
    test_samples = scan_dataset(args.test_root) if args.test_root and os.path.exists(args.test_root) else None

    print(f"Extract train features: {len(train_samples)}")
    train_pack = extract_features(train_samples, preprocess, model, device, args.batch_size, args.num_workers)
    bank = build_balanced_bank(train_pack, max_per_model_class=args.max_per_model_class, seed=args.seed)
    print(f"Bank real={bank['real_features'].shape[0]} fake={bank['fake_features'].shape[0]}")

    print("Compute train leave-one-out NN scores")
    sr, sf, sc = compute_scores(train_pack, bank, k=args.k, device=device, batch_size=args.batch_size, exclude_self=True)
    write_scores_csv(os.path.join(args.output_dir, "clip_nn_train.csv"), train_pack, sr, sf, sc, args.tau, args.bias)

    print(f"Extract val features: {len(val_samples)}")
    val_pack = extract_features(val_samples, preprocess, model, device, args.batch_size, args.num_workers)
    sr, sf, sc = compute_scores(val_pack, bank, k=args.k, device=device, batch_size=args.batch_size, exclude_self=False)
    write_scores_csv(os.path.join(args.output_dir, "clip_nn_val.csv"), val_pack, sr, sf, sc, args.tau, args.bias)

    if test_samples is not None:
        print(f"Extract test features: {len(test_samples)}")
        test_pack = extract_features(test_samples, preprocess, model, device, args.batch_size, args.num_workers)
        sr, sf, sc = compute_scores(test_pack, bank, k=args.k, device=device, batch_size=args.batch_size, exclude_self=False)
        write_scores_csv(os.path.join(args.output_dir, "clip_nn_test.csv"), test_pack, sr, sf, sc, args.tau, args.bias)

    print({
        "train_csv": os.path.join(args.output_dir, "clip_nn_train.csv"),
        "val_csv": os.path.join(args.output_dir, "clip_nn_val.csv"),
        "test_csv": os.path.join(args.output_dir, "clip_nn_test.csv") if test_samples is not None else None,
    })


if __name__ == "__main__":
    main()
