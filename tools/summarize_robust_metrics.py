#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
from pathlib import Path

root = Path(__import__("os").environ.get("RESULT_ROOT", "./results/robust_fusion_clipnn_vitl14_k10_m500"))

names = [
    "clean",
    "jpeg95",
    "jpeg75",
    "jpeg50",
    "resize075",
    "resize05",
    "blur3",
    "blur5",
    "noise5",
    "noise10",
    "crop09",
    "crop075",
]

def find_metric_dict(d):
    # 兼容不同 metrics.json 的保存格式
    candidate_keys = [
        "test_fixed05_metrics",
        "fixed05_metrics",
        "test_metrics",
        "metrics",
        "test",
        "fixed05",
    ]

    for k in candidate_keys:
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            return d[k]

    # 有些文件可能直接是指标字典
    return d

def get_metric(m, names, default=""):
    for n in names:
        if n in m:
            return m[n]
    return default

rows = []

for name in names:
    path = root / name / "metrics.json"

    row = {
        "name": name,
        "AP": "",
        "Accuracy": "",
        "RealAccuracy": "",
        "FakeAccuracy": "",
        "MacroAccuracy": "",
        "MacroBalancedAcc": "",
        "WorstModelAccuracy": "",
        "metrics_path": str(path),
    }

    if not path.exists():
        row["AP"] = "MISSING"
        rows.append(row)
        continue

    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        row["AP"] = f"READ_ERROR:{e}"
        rows.append(row)
        continue

    m = find_metric_dict(d)

    row["AP"] = get_metric(m, ["AP", "ap", "average_precision", "AveragePrecision"])
    row["Accuracy"] = get_metric(m, ["Accuracy", "accuracy", "acc"])
    row["RealAccuracy"] = get_metric(m, ["RealAccuracy", "real_accuracy", "real_acc"])
    row["FakeAccuracy"] = get_metric(m, ["FakeAccuracy", "fake_accuracy", "fake_acc"])
    row["MacroAccuracy"] = get_metric(m, ["MacroAccuracy", "macro_accuracy", "macro_acc"])
    row["MacroBalancedAcc"] = get_metric(m, ["MacroBalancedAcc", "macro_balanced_acc", "macro_balanced_accuracy"])
    row["WorstModelAccuracy"] = get_metric(m, ["WorstModelAccuracy", "worst_model_accuracy", "worst_group_accuracy", "WorstGroupAccuracy"])

    rows.append(row)

out_csv = root / "robust_summary.csv"

with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"[DONE] summary saved to: {out_csv}")
for r in rows:
    print(
        f"{r['name']:10s} "
        f"AP={r['AP']} "
        f"Acc={r['Accuracy']} "
        f"Real={r['RealAccuracy']} "
        f"Fake={r['FakeAccuracy']}"
    )
