from pathlib import Path
import json
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compact_checkpoint_metadata():
    s = torch.load(ROOT / "weights/semantic_best.pth", map_location="cpu", weights_only=True)
    a = torch.load(ROOT / "weights/artifact_best.pth", map_location="cpu", weights_only=True)
    f = torch.load(ROOT / "weights/fusion_best.pth", map_location="cpu", weights_only=True)

    assert s["checkpoint_type"] == "semantic_delta"
    assert s["base_clip_model"] == "ViT-L/14"
    assert s["metadata"]["epoch"] == 5

    assert a["checkpoint_type"] == "artifact_inference"
    assert a["metadata"]["epoch"] == 20
    assert a["metadata"]["artifact_build_meta"]["image_size"] == 224
    assert a["metadata"]["artifact_build_meta"]["npr_scales"] == [0.25, 0.5, 0.75]

    assert f["checkpoint_type"] == "rider_fusion_delta"
    assert f["epoch"] == 4
    assert f["metadata"]["load_order"][-1] == "fusion_best.pth"


def test_manifest_audit_counts():
    audit = json.load(open(ROOT / "datasets/manifests/dataset_audit.json", encoding="utf-8"))
    assert audit["num_images"] == 284568
    assert audit["split_counts"] == {"train": 159987, "val": 3197, "test": 121384}


def test_results_directory_is_compact():
    files = sorted(p.name for p in (ROOT / "results").iterdir() if p.is_file())
    assert files == ["ablation_results.csv", "main_results.csv", "robustness_results.csv"]
    assert not any(p.is_dir() for p in (ROOT / "results").iterdir())


def test_ablation_configs_live_under_experiments():
    assert (ROOT / "experiments/semantic_ablation").is_dir()
    assert (ROOT / "experiments/artifact_ablation").is_dir()


def test_public_repository_layout():
    assert not (ROOT / "GITHUB_UPLOAD.md").exists()
    assert not (ROOT / "RELEASE_CHECKLIST.md").exists()
    assert not (ROOT / "requirements-optional.txt").exists()
    assert (ROOT / ".gitattributes").is_file()
    assert (ROOT / "scripts/04_train_rider_from_scratch.sh").is_file()
    assert (ROOT / "scripts/06_prepare_robustness.sh").is_file()
    assert (ROOT / "tools/inspect_checkpoints.py").is_file()


def test_final_config_matches_key_reported_settings():
    cfg = yaml.safe_load((ROOT / "configs/final_config.yaml").read_text(encoding="utf-8"))
    assert cfg["semantic"]["epochs"] == 5
    assert cfg["semantic"]["finetune_last_n"] == 1
    assert cfg["semantic"]["patch_shuffle_prob"] == 0.0
    assert cfg["artifact"]["branches"] == ["spectral_mag", "wavelet", "npr"]
    assert cfg["artifact"]["npr_scales"] == [0.25, 0.5, 0.75]
    assert cfg["fusion"]["selected_epoch"] == 4
    assert cfg["fusion"]["artifact_failure_prob"] == 0.35
    assert cfg["fusion"]["train_sampler"] == "model_balanced"
    assert cfg["clip_nn"]["k"] == 10
    assert cfg["robustness"]["num_settings"] == 11


def test_reference_environment_and_core_requirements():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    env = (ROOT / "environment_reference.txt").read_text(encoding="utf-8")

    assert "albumentations==2.0.6" in req
    assert "openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1" in req
    assert "torch==" not in req
    assert "torchvision==" not in req

    assert "Python==3.12.0" in env
    assert "torch==2.7.1+cu118" in env
    assert "torchvision==0.22.1+cu118" in env
    assert "CUDA_runtime==11.8" in env


def test_server_directory_counts_match_clean_split_totals():
    import csv

    counts_path = ROOT / "datasets/manifests/dataset_directory_counts.csv"
    totals = {"train": 0, "val": 0, "test": 0}
    with counts_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            directory = row["directory"]
            n = int(row["num_images"])
            for split in totals:
                if directory.startswith(f"dataset/{split}/"):
                    totals[split] += n
                    break

    assert totals == {"train": 159987, "val": 3197, "test": 121384}
    assert (ROOT / "datasets/dataset_layout.txt").is_file()
