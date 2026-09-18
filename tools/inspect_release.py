#!/usr/bin/env python3
from pathlib import Path
import torch, json

for name in ["semantic_best.pth","artifact_best.pth","fusion_best.pth"]:
    p=Path("weights")/name
    x=torch.load(p,map_location="cpu",weights_only=True)
    print("\n",name, f"{p.stat().st_size/1024/1024:.2f} MiB")
    print("checkpoint_type =", x.get("checkpoint_type"))
    if "epoch" in x: print("epoch =",x["epoch"])
    if isinstance(x.get("metadata"),dict): print(json.dumps(x["metadata"],indent=2,default=str))
