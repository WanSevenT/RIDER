from pathlib import Path
from typing import Dict, Union
import torch

PathLike = Union[str, Path]

def _load(path: PathLike, device: str):
    return torch.load(str(path), map_location=device)

def load_pretrained_rider(model, weights_dir: PathLike = "weights", device: str = "cpu") -> Dict[str, object]:
    """Load the compact RIDER release checkpoints in the required order.

    The model must already be constructed with the base OpenAI CLIP ViT-L/14.
    Loading order is semantically important because fusion_best.pth contains
    Phase-II-updated artifact BatchNorm buffers.
    """
    w = Path(weights_dir)
    semantic = _load(w / "semantic_best.pth", device)
    artifact = _load(w / "artifact_best.pth", device)
    fusion = _load(w / "fusion_best.pth", device)

    if semantic.get("base_clip_model") not in {None, "ViT-L/14"}:
        raise ValueError(f"Expected ViT-L/14 base CLIP, got {semantic.get('base_clip_model')}")

    model.clip_model.load_state_dict(semantic["clip_delta_state_dict"], strict=False)
    model.semantic_classifier.load_state_dict(semantic["semantic_classifier_state_dict"], strict=True)
    model.artifact_extractor.load_state_dict(artifact["artifact_extractor_state_dict"], strict=True)
    model.artifact_classifier.load_state_dict(artifact["artifact_classifier_state_dict"], strict=True)
    missing, unexpected = model.load_state_dict(fusion["model_state_dict"], strict=False)
    return {
        "semantic": str(w / "semantic_best.pth"),
        "artifact": str(w / "artifact_best.pth"),
        "fusion": str(w / "fusion_best.pth"),
        "fusion_missing": list(missing),
        "fusion_unexpected": list(unexpected),
        "load_order": ["OpenAI CLIP ViT-L/14", "semantic_best.pth", "artifact_best.pth", "fusion_best.pth"],
    }
