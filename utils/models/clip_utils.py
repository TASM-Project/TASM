import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

def interpolate_clip_position_embeddings(clip_model: CLIPModel, image_size: int):
    """Resize HF CLIP ViT absolute position embeddings for larger square inputs."""
    vision_config = clip_model.config.vision_config
    patch_size = int(vision_config.patch_size)
    old_image_size = int(vision_config.image_size)
    
    if image_size == old_image_size:
        return clip_model
    if image_size % patch_size != 0:
        raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")

    emb = clip_model.vision_model.embeddings
    old_weight = emb.position_embedding.weight.data
    num_pos, dim = old_weight.shape
    cls_pos = old_weight[:1]
    patch_pos = old_weight[1:]
    old_grid = int(math.sqrt(patch_pos.shape[0]))
    new_grid = image_size // patch_size

    if old_grid * old_grid != patch_pos.shape[0]:
        raise RuntimeError(f"Cannot infer old grid from {patch_pos.shape[0]} patch positions")

    patch_pos = patch_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_grid * new_grid, dim)
    new_weight = torch.cat([cls_pos, patch_pos], dim=0)

    new_num_pos = new_weight.shape[0]
    new_embedding = nn.Embedding(new_num_pos, dim).to(old_weight.device)
    new_embedding.weight.data.copy_(new_weight)
    emb.position_embedding = new_embedding
    emb.register_buffer("position_ids", torch.arange(new_num_pos, device=old_weight.device).expand((1, -1)), persistent=False)

    vision_config.image_size = image_size
    emb.image_size = image_size
    print(f"[CLIP] Interpolated position embeddings: {num_pos} -> {new_num_pos} tokens, image_size {old_image_size} -> {image_size}")
    
    return clip_model