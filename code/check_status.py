# -*- coding: utf-8 -*-
# 体检脚本：评估 baseline 线性头的效果 + 核对提交 CSV 格式
import os
import numpy as np
import torch

CACHE = r"D:\aicomp\cache"
OUT = r"D:\aicomp\output"

feats = torch.from_numpy(np.load(os.path.join(CACHE, "train_feats_all.npy")))
labels = torch.from_numpy(np.load(os.path.join(CACHE, "train_labels_all.npy")))
sd = torch.load(os.path.join(CACHE, "linear_all.pt"), map_location="cpu")
w, b = sd["weight"], sd["bias"]
print("feats", tuple(feats.shape), "labels", tuple(labels.shape))
print("weight", tuple(w.shape), "bias", tuple(b.shape))

logits = feats @ w.T + b
pred = logits.argmax(1)
acc = (pred == labels).float().mean().item()
print("train_acc (linear_all) = %.4f" % acc)

vals, counts = np.unique(labels.numpy(), return_counts=True)
print("classes in train:", len(vals), "| samples/class min=%d max=%d mean=%.1f"
      % (counts.min(), counts.max(), counts.mean()))

csv = os.path.join(OUT, "pred_results.csv")
with open(csv, encoding="utf-8") as f:
    lines = f.read().strip().splitlines()
print("csv rows:", len(lines))
for l in lines[:5]:
    print("  ", repr(l))
