#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def to_rgb(path):
    return Image.open(path).convert("RGB")

def resize_attack(img, scale):
    w, h = img.size
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img2 = img.resize((nw, nh), Image.BICUBIC)
    return img2.resize((w, h), Image.BICUBIC)

def noise_attack(img, sigma):
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def crop_attack(img, ratio):
    w, h = img.size
    nw, nh = int(w * ratio), int(h * ratio)
    nw = max(1, min(w, nw))
    nh = max(1, min(h, nh))
    left = (w - nw) // 2
    top = (h - nh) // 2
    cropped = img.crop((left, top, left + nw, top + nh))
    return cropped.resize((w, h), Image.BICUBIC)

def process(src_path, dst_path, corruption, strength):
    img = to_rgb(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if corruption == "jpeg":
        # JPEG outputs are normalized to the .jpg suffix
        img.save(dst_path.with_suffix(".jpg"), quality=int(strength), subsampling=2)
        return

    if corruption == "resize":
        img = resize_attack(img, float(strength))
    elif corruption == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=float(strength)))
    elif corruption == "noise":
        img = noise_attack(img, float(strength))
    elif corruption == "crop":
        img = crop_attack(img, float(strength))
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    # Non-JPEG perturbations keep the original suffix
    img.save(dst_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True)
    parser.add_argument("--dst_root", required=True)
    parser.add_argument("--corruption", required=True,
                        choices=["jpeg", "resize", "blur", "noise", "crop"])
    parser.add_argument("--strength", required=True)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional NumPy RNG seed for deterministic Gaussian-noise generation")
    parser.add_argument("--include_top_dirs", nargs="*", default=None,
                        help="Optional top-level directories to include under --src_root")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    count = 0
    skipped = 0
    filtered = 0
    include_top_dirs = set(args.include_top_dirs or [])

    paths = src_root.rglob("*")
    if args.seed is not None:
        paths = sorted(paths)

    for src_path in paths:
        if src_path.suffix.lower() not in IMG_EXTS:
            continue

        rel = src_path.relative_to(src_root)
        if include_top_dirs and (not rel.parts or rel.parts[0] not in include_top_dirs):
            filtered += 1
            continue
        dst_path = dst_root / rel

        if args.corruption == "jpeg":
            final_path = dst_path.with_suffix(".jpg")
        else:
            final_path = dst_path

        if args.skip_existing and final_path.exists():
            skipped += 1
            continue

        process(src_path, dst_path, args.corruption, args.strength)
        count += 1

    seed_msg = "none" if args.seed is None else str(args.seed)
    print(f"[DONE] {args.corruption}-{args.strength}: processed={count}, skipped={skipped}, filtered={filtered}, seed={seed_msg}, dst={dst_root}")

if __name__ == "__main__":
    main()
