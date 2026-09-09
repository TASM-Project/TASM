#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

# ---------------------------------------------------------
# Shared Constants (Purified for TASM 6-Bin Unified Logic)
# ---------------------------------------------------------
ALL_ATTRS = [
    "BalancingElements", "ColorHarmony", "Content", "DoF", "Light",
    "MotionBlur", "Object", "Repetition", "RuleOfThirds", "Symmetry", "VividColor"
]


class ProjectionHead(nn.Module):
    """Projection head used during Stage 1 Text-guided Adaptation."""

    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TASM_AADB(nn.Module):
    """Stage 2: Multi-task network for TASM Framework."""

    def __init__(self, clip_model: CLIPModel, feat_dim=512, hidden_dim=512, proto_dim=512, dropout=0.3):
        super().__init__()
        self.clip_model = clip_model
        self.shared = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_attr = nn.Linear(hidden_dim, len(ALL_ATTRS))
        self.head_score = nn.Linear(hidden_dim, 1)

        # Attribute query branch for Micro-Level Disentanglement
        self.attr_queries = nn.Parameter(torch.randn(len(ALL_ATTRS), hidden_dim) * 0.02)
        self.attr_proto_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proto_dim),
        )

    def forward(self, images):
        feats = self.clip_model.get_image_features(pixel_values=images).float()
        h = self.shared(feats)
        attrs = torch.tanh(self.head_attr(h))
        score = torch.sigmoid(self.head_score(h)).squeeze(-1)

        attr_h = h.unsqueeze(1) + self.attr_queries.unsqueeze(0)
        # 归一化后的视觉特定属性特征 (对应论文架构图中的 u_{i,k})
        attr_proto_emb = F.normalize(self.attr_proto_proj(attr_h), dim=-1)
        return attrs, score, attr_proto_emb


class TASM_AVA(nn.Module):
    """Stage 2: Aesthetic distribution regressor for AVA."""

    def __init__(self, clip_model: CLIPModel, feat_dim=512, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.clip_model = clip_model
        self.shared = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_dist = nn.Linear(hidden_dim, 10)
        score_values = torch.linspace(0.0, 1.0, steps=10)
        self.register_buffer("score_values", score_values)

    def forward(self, images):
        feat = self.clip_model.get_image_features(pixel_values=images).float()
        h = self.shared(feat)
        logits = self.head_dist(h)
        prob = F.softmax(logits, dim=-1)
        score_norm = (prob * self.score_values.view(1, -1)).sum(dim=-1)
        score_raw = score_norm * 9.0 + 1.0
        return logits, prob, score_norm, score_raw