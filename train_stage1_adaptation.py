#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASM Stage 1: Macro-Level Aesthetic Adaptation
"""
import argparse
import csv
import json
import math
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.transforms import build_transform
from utils.data_utils import image_file_from_item
from models.tasm import ProjectionHead, ALL_ATTRS
from models.losses import standard_symmetric_info_nce


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_texts(x) -> list:
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    if isinstance(x, list):
        return [t.strip() for t in x if isinstance(t, str) and t.strip()]
    if isinstance(x, dict):
        return clean_texts(x.get("texts", []))
    return []


class AADBTextStage1Dataset(Dataset):
    def __init__(self, jsonl_file: str, image_root: str, score_csv: str, transform, min_texts: int = 1):
        self.image_root = image_root
        self.transform = transform
        self.score_map = {}
        with open(score_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.score_map[row["ImageFile"]] = row

        self.items = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                img_file = image_file_from_item(item)
                if img_file not in self.score_map: continue
                attrs = item.get("attributes", {})
                valid = []
                for attr in ALL_ATTRS:
                    texts = clean_texts(attrs.get(attr))
                    if len(texts) >= min_texts:
                        valid.append((attr, texts))
                if valid:
                    self.items.append({"image_file": img_file, "valid": valid})
        print(f"[DATA] Stage1 valid images: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img_file = item["image_file"]
        image = Image.open(os.path.join(self.image_root, img_file)).convert("RGB")
        image = self.transform(image)

        attr, texts = random.choice(item["valid"])
        text = random.choice(texts)
        return image, text


def main():
    parser = argparse.ArgumentParser(description="TASM Stage 1: Macro-Level Adaptation (Standard InfoNCE)")
    parser.add_argument("--jsonl_train", required=True, help="Repaired AADB 6-bin JSONL, train only.")
    parser.add_argument("--score_csv_train", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--clip_path", required=True)
    parser.add_argument("--out_dir", default="./weights/stage1")
    parser.add_argument("--save_name", default="tasm_stage1.pth")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--preserve_full_frame", action="store_true")
    parser.add_argument("--lr_backbone", type=float, default=5e-6)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    transform = build_transform(args.image_size, preserve_full_frame=args.preserve_full_frame, train=True)
    dataset = AADBTextStage1Dataset(args.jsonl_train, args.image_dir, args.score_csv_train, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                        drop_last=True)

    clip_model = CLIPModel.from_pretrained(args.clip_path).to(args.device)
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_path)

    img_dim = txt_dim = clip_model.config.projection_dim
    img_head = ProjectionHead(img_dim, 256).to(args.device)
    txt_head = ProjectionHead(txt_dim, 256).to(args.device)

    decay, no_decay = [], []
    for name, p in clip_model.named_parameters():
        if not p.requires_grad: continue
        if name.endswith("bias") or "layer_norm" in name.lower() or "ln_" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay, "lr": args.lr_backbone, "weight_decay": args.weight_decay},
        {"params": no_decay, "lr": args.lr_backbone, "weight_decay": 0.0},
        {"params": img_head.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": txt_head.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay},
    ])

    best_loss = math.inf
    for epoch in range(args.epochs):
        clip_model.train();
        img_head.train();
        txt_head.train()
        pbar = tqdm(loader, desc=f"Stage 1 Epoch {epoch + 1}/{args.epochs}")
        total_loss = 0.0

        for images, texts in pbar:
            images = images.to(args.device)
            tok = tokenizer(list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt").to(
                args.device)

            img_proj = img_head(clip_model.get_image_features(pixel_values=images))
            txt_proj = txt_head(clip_model.get_text_features(**tok))

            loss = standard_symmetric_info_nce(img_proj, txt_proj, temperature=args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(clip_model.parameters()) + list(img_head.parameters()) + list(txt_head.parameters()), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"[EPOCH {epoch + 1}] avg_loss={avg_loss:.6f}")

        state = {"clip": clip_model.state_dict(), "args": vars(args)}
        torch.save(state, os.path.join(args.out_dir, args.save_name.replace(".pth", "_last.pth")))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state, os.path.join(args.out_dir, args.save_name))


if __name__ == "__main__":
    main()