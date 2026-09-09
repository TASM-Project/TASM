#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image, ImageFile
from transformers import CLIPModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.transforms import build_transform
from utils.metrics import safe_srcc, safe_plcc
from utils.data_utils import get_attr_value
from models.tasm import TASM_AADB, ALL_ATTRS


class AADBTestDataset(Dataset):
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = self.transform(Image.open(os.path.join(self.image_root, item["image_file"])).convert("RGB"))
        return image, torch.tensor(item["attrs"], dtype=torch.float32), torch.tensor(item["score"], dtype=torch.float32), item["image_file"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--score_csv_test", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--clip_path", required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    clip_model = CLIPModel.from_pretrained(args.clip_path).to(args.device)
    model = TASM_AADB(clip_model).to(args.device)
    
    model.clip_model.load_state_dict(ckpt["clip"], strict=True)
    model.shared.load_state_dict(ckpt["shared"], strict=True)
    model.head_attr.load_state_dict(ckpt["head_attr"], strict=True)
    model.head_score.load_state_dict(ckpt["head_score"], strict=True)
    model.eval()

    loader = DataLoader(AADBTestDataset(args.score_csv_test, args.image_dir, build_transform(args.image_size)), batch_size=64, num_workers=8)

    pred_a, gt_a, pred_s, gt_s = [], [], [], []
    with torch.no_grad():
        for images, attrs, score, _ in tqdm(loader, desc="Testing"):
            pa, ps, _ = model(images.to(args.device))
            pred_a.append(pa.cpu().numpy()); gt_a.append(attrs.numpy())
            pred_s.extend(ps.cpu().numpy()); gt_s.extend(score.numpy())

    pred_a, gt_a = np.concatenate(pred_a, axis=0), np.concatenate(gt_a, axis=0)
    pred_s, gt_s = np.asarray(pred_s), np.asarray(gt_s)

    print("\n================ SCORE =================")
    print(f"Overall SRCC: {safe_srcc(pred_s, gt_s):.4f}")
    print(f"Overall PLCC: {safe_plcc(pred_s, gt_s):.4f}")
    print("\n============= MEAN ATTRIBUTE =============")
    print(f"Mean SRCC: {np.mean([safe_srcc(pred_a[:, i], gt_a[:, i]) for i in range(len(ALL_ATTRS))]):.4f}")

if __name__ == "__main__":
    main()