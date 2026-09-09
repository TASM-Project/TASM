import torch
import torch.nn as nn
import torch.nn.functional as F


def standard_symmetric_info_nce(img_features, txt_features, temperature=0.07):
    """
    Stage 1: Standard Symmetric InfoNCE Loss (L_align) for Macro-Level Adaptation.
    Aligns global visual representations with score-conditioned linguistic priors.
    """
    img_features = F.normalize(img_features, dim=-1)
    txt_features = F.normalize(txt_features, dim=-1)

    logits_per_image = img_features @ txt_features.t() / temperature
    logits_per_text = txt_features @ img_features.t() / temperature

    labels = torch.arange(img_features.size(0), device=img_features.device)

    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2.0


def spectrum_matching_loss(attr_emb, attrs_gt, proto_tensor, temperature=0.07, sigma=1.0):
    """
    Stage 2: 6-Bin Spectrum Matching Loss (L_SM) via KL Divergence.
    Maps continuous scores [-1, 1] to a 6-bin target distribution using Gaussian softening,
    and forces the visual-text cosine similarities to match it.
    """
    if proto_tensor is None:
        return attr_emb.new_tensor(0.0)

    B, A, D = attr_emb.shape
    losses = []

    # 1. 映射分数 [-1, 1] 到索引 [0, 5]
    target_idx = (attrs_gt + 1.0) / 2.0 * 5.0  # [B, A]
    bins = torch.arange(6, device=attr_emb.device, dtype=torch.float32).view(1, 1, 6)

    # 2. Gaussian Softening 生成目标分布 t_{i,k}
    diff_sq = (bins - target_idx.unsqueeze(-1)) ** 2
    target_dist = F.softmax(-diff_sq / (2.0 * sigma ** 2), dim=-1)  # [B, A, 6]

    for a in range(A):
        # 3. 计算特定属性特征与 6 个锚点的余弦相似度 s_{i,k}
        logits = attr_emb[:, a, :] @ proto_tensor[a].t() / temperature  # [B, 6]
        log_probs = F.log_softmax(logits, dim=-1)

        # 4. KL 散度约束
        loss_a = F.kl_div(log_probs, target_dist[:, a, :], reduction='batchmean')
        losses.append(loss_a)

    return torch.stack(losses).mean()


def distribution_emd_loss(prob, target_dist, eps=1e-8):
    """Earth Mover's Distance Loss for absolute score distributions."""
    pred_cdf = torch.cumsum(prob, dim=-1)
    target_cdf = torch.cumsum(target_dist, dim=-1)
    return torch.sqrt(torch.mean((pred_cdf - target_cdf) ** 2, dim=-1) + eps).mean()
