# -*- coding: utf-8 -*-
# CLIP ViT-B/32 frozen features + linear classifier (first baseline)
# Usage:
#   python baseline.py --per-class 5      quick test (~5 min)
#   python baseline.py                    full run (~20-40 min)

import os, sys, json, time, random, argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

TRAIN_DIR  = r"D:\aicomp\train"
TEST_DIR   = r"D:\aicomp\test"
OUT_DIR    = r"D:\aicomp\output"
CACHE_DIR  = r"D:\aicomp\cache"
MODEL_NAME = "openai/clip-vit-base-patch32"
IMG_EXT    = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def list_train_samples(per_class=0, seed=42):
    classes = sorted([d.name for d in Path(TRAIN_DIR).iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = random.Random(seed)
    samples = []
    for c in classes:
        files = sorted([f for f in Path(TRAIN_DIR, c).iterdir() if f.suffix.lower() in IMG_EXT])
        if per_class > 0 and len(files) > per_class:
            files = rng.sample(files, per_class)
        for f in files:
            samples.append((str(f), class_to_idx[c]))
    return samples, classes


def list_test_samples():
    files = sorted([f for f in Path(TEST_DIR).iterdir() if f.suffix.lower() in IMG_EXT])
    return [str(f) for f in files]


class ClipImageDataset(Dataset):
    def __init__(self, paths, labels, processor):
        self.paths, self.labels, self.processor = paths, labels, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        px = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return px, self.labels[idx]


def get_image_features(model, pixel_values):
    out = model.get_image_features(pixel_values=pixel_values)
    if isinstance(out, torch.Tensor):
        return out
    if getattr(out, "image_embeds", None) is not None:
        return out.image_embeds
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    raise RuntimeError("cannot extract image features from %s" % type(out))


@torch.no_grad()
def extract_features(model, dataset, device, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    feats, labels = [], []
    for px, y in tqdm(loader, desc="extract", ncols=80):
        px = px.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            f = get_image_features(model, px)
        feats.append(F.normalize(f.float(), dim=-1).cpu())
        labels.append(y)
    return torch.cat(feats, 0), torch.cat(labels, 0)


def train_linear(feats, labels, num_classes, device, epochs, lr, batch_size):
    feats, labels = feats.to(device), labels.to(device)
    clf = nn.Linear(feats.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    n = feats.shape[0]
    t0 = time.time()
    for ep in range(epochs):
        clf.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(clf(feats[idx]), labels[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        if (ep + 1) % 10 == 0 or ep == 0:
            clf.eval()
            with torch.no_grad():
                acc = (clf(feats).argmax(1) == labels).float().mean().item()
            print("epoch %3d  loss %.4f  train_acc %.4f  (%.0fs)"
                  % (ep + 1, total / n, acc, time.time() - t0))
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=0, help="images per class, 0 = all")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--extract-bs", type=int, default=64)
    ap.add_argument("--train-bs", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    from transformers import CLIPModel, CLIPProcessor
    print("loading model ...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    tag = "all" if args.per_class == 0 else "pc%d" % args.per_class
    ftr = os.path.join(CACHE_DIR, "train_feats_%s.npy" % tag)
    ltr = os.path.join(CACHE_DIR, "train_labels_%s.npy" % tag)
    fte = os.path.join(CACHE_DIR, "test_feats.npy")
    nte = os.path.join(CACHE_DIR, "test_names.json")

    samples, classes = list_train_samples(args.per_class)
    print("train samples:", len(samples), " classes:", len(classes))

    if os.path.exists(ftr) and os.path.exists(ltr) and not args.no_cache:
        print("loading cached train features ...")
        train_feats = torch.from_numpy(np.load(ftr))
        train_labels = torch.from_numpy(np.load(ltr))
    else:
        ds = ClipImageDataset([s[0] for s in samples], [s[1] for s in samples], processor)
        train_feats, train_labels = extract_features(model, ds, device, args.extract_bs, args.workers)
        np.save(ftr, train_feats.numpy()); np.save(ltr, train_labels.numpy())
        print("train features:", tuple(train_feats.shape))

    print("training linear classifier ...")
    clf = train_linear(train_feats, train_labels, len(classes), device,
                       args.epochs, args.lr, args.train_bs)
    torch.save(clf.state_dict(), os.path.join(CACHE_DIR, "linear_%s.pt" % tag))

    test_paths = list_test_samples()
    print("test images:", len(test_paths))
    if os.path.exists(fte) and os.path.exists(nte) and not args.no_cache:
        print("loading cached test features ...")
        test_feats = torch.from_numpy(np.load(fte))
        test_names = json.load(open(nte, encoding="utf-8"))
    else:
        ds = ClipImageDataset(test_paths, [-1] * len(test_paths), processor)
        test_feats, _ = extract_features(model, ds, device, args.extract_bs, args.workers)
        np.save(fte, test_feats.numpy())
        test_names = [os.path.basename(p) for p in test_paths]
        json.dump(test_names, open(nte, "w", encoding="utf-8"))
        print("test features:", tuple(test_feats.shape))

    clf.eval()
    with torch.no_grad():
        preds = clf(test_feats.to(device)).argmax(1).cpu().numpy()

    out_csv = os.path.join(OUT_DIR, "pred_results.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        for name, p in zip(test_names, preds):
            f.write("%s, %04d\n" % (name, int(p)))
    print("saved:", out_csv, " rows:", len(preds))


if __name__ == "__main__":
    main()
