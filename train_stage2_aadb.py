#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASM Stage 2: Micro-Level Disentanglement
"""
import argparse
import csv
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image, ImageFile
from transformers import CLIPModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.transforms import build_transform
from utils.data_utils import get_attr_value
from models.tasm import TASM_AADB, ALL_ATTRS
from models.losses import spectrum_matching_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prototype_tensor(bank_path, device):
    if not bank_path: return None
    vectors = torch.load(bank_path, map_location="cpu")["prototype_vectors"]
    D = next(iter(vectors.values())).numel()

    proto = torch.zeros(len(ALL_ATTRS), 6, D, dtype=torch.float32)
    SEMANTIC_LEVELS = ["Very Bad", "Bad", "Slightly Bad", "Slightly Good", "Good", "Very Good"]

    for i, attr in enumerate(ALL_ATTRS):
        for j, level in enumerate(SEMANTIC_LEVELS):
            key = f"{attr}||{level}"
            if key in vectors:
                proto[i, j] = F.normalize(vectors[key].float(), dim=0)
            else:
                raise ValueError(f"CRITICAL ERROR: Missing prototype anchor '{key}' in bank!")
    return proto.to(device)


class AADBStage2Dataset(Dataset):
    def __init__(self, csv_file, image_root, transform):
        self.image_root = image_root
        self.transform = transform
        self.items = []
        with open(csv_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.items.append({
                    "image_file": row["ImageFile"],
                    "attrs": [get_attr_value(row, a) for a in ALL_ATTRS],
                    "score": float(row["score"])
                })

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = self.transform(Image.open(os.path.join(self.image_root, item["image_file"])).convert("RGB"))
        return image, torch.tensor(item["attrs"], dtype=torch.float32), torch.tensor(item["score"], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="TASM Stage 2: Spectrum Matching (Full Train Data)")
    parser.add_argument("--score_csv_train", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--clip_path", required=True)
    parser.add_argument("--stage1_ckpt", default=None)
    parser.add_argument("--semantic_spectrum", default=None)
    parser.add_argument("--out_dir", default="./weights/stage2_aadb")
    parser.add_argument("--save_prefix", default="tasm_aadb")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    transform = build_transform(args.image_size, train=True)

    train_dataset = AADBStage2Dataset(args.score_csv_train, args.image_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    print(f"[INFO] Training with 100% of the dataset: {len(train_dataset)} images.")

    clip_model = CLIPModel.from_pretrained(args.clip_path).to(args.device)
    if args.stage1_ckpt:
        clip_model.load_state_dict(torch.load(args.stage1_ckpt, map_location="cpu").get("clip", {}), strict=False)

    proto_tensor = load_prototype_tensor(args.semantic_spectrum, args.device) if args.semantic_spectrum else None
    proto_dim = proto_tensor.size(-1) if proto_tensor is not None else 512

    model = TASM_AADB(clip_model, feat_dim=clip_model.config.projection_dim, proto_dim=proto_dim).to(args.device)

    optimizer = torch.optim.AdamW([
        {"params": model.clip_model.parameters(), "lr": 1e-6},
        {"params": [p for n, p in model.named_parameters() if "clip" not in n], "lr": 1e-4},
    ], weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage 2 Epoch {epoch}/{args.epochs}")
        for images, attrs_gt, score_gt in pbar:
            images = images.to(args.device)
            attrs_gt = attrs_gt.to(args.device)
            score_gt = score_gt.to(args.device)

            attrs_pred, score_pred, attr_emb = model(images)
            loss_s = F.mse_loss(score_pred, score_gt)
            loss_a = F.mse_loss(attrs_pred, attrs_gt)

            loss_sm = spectrum_matching_loss(attr_emb, attrs_gt,
                                             proto_tensor) if proto_tensor is not None else score_pred.new_tensor(0.0)

            loss = loss_s + loss_a + loss_sm

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}", sc=f"{loss_s.item():.4f}", at=f"{loss_a.item():.4f}",
                             sm=f"{loss_sm.item():.4f}")

        if epoch == args.epochs:
            state_dict = {
                "clip": model.clip_model.state_dict(),
                "shared": model.shared.state_dict(),
                "head_attr": model.head_attr.state_dict(),
                "head_score": model.head_score.state_dict(),
                "attr_queries": model.attr_queries.detach().cpu(),
                "attr_proto_proj": model.attr_proto_proj.state_dict(),
                "args": vars(args)
            }
            save_path = os.path.join(args.out_dir, f"{args.save_prefix}_last.pth")
            torch.save(state_dict, save_path)
            print(f"[INFO] Training complete. Final model saved to {save_path}")


if __name__ == "__main__":
    main()