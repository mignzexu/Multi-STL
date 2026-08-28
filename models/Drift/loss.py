import math

import torch
import torch.nn.functional as F
from torch import nn


def _finite_differences(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def _limit_spatial_tokens(feature, max_tokens):
    h, w = feature.shape[-2:]
    tokens = h * w
    if max_tokens <= 0 or tokens <= max_tokens:
        return feature

    scale = math.sqrt(max_tokens / tokens)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    while new_h * new_w > max_tokens:
        if new_h >= new_w and new_h > 1:
            new_h -= 1
        elif new_w > 1:
            new_w -= 1
        else:
            break
    return F.adaptive_avg_pool2d(feature, (new_h, new_w))


def _feature_pyramid(x, max_tokens):
    features = []
    current = x
    for _ in range(3):
        limited = _limit_spatial_tokens(current, max_tokens)
        features.append(limited.flatten(1))
        dx, dy = _finite_differences(limited)
        features.append(dx.flatten(1))
        features.append(dy.flatten(1))
        if current.shape[-2] < 2 or current.shape[-1] < 2:
            break
        current = F.avg_pool2d(current, kernel_size=2, stride=2)
    return features


def _cdist(x, y, eps=1e-8):
    x_dot_y = torch.einsum("bnd,bmd->bnm", x, y)
    x_norm = torch.einsum("bnd,bnd->bn", x, x)
    y_norm = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = x_norm[:, :, None] + y_norm[:, None, :] - 2 * x_dot_y
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def drift_sample_loss(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_neg=None,
    r_list=(0.02, 0.05, 0.2),
):
    batch, gen_count, feature_dim = gen.shape
    pos_count = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen[:, :0, :]
    neg_count = fixed_neg.shape[1]

    old_gen = gen.detach()
    fixed_pos = fixed_pos.detach()
    fixed_neg = fixed_neg.detach()

    weight_gen = gen.new_ones(batch, gen_count)
    weight_pos = gen.new_ones(batch, pos_count)
    if weight_neg is None:
        weight_neg = gen.new_ones(batch, neg_count)
    elif not torch.is_tensor(weight_neg):
        weight_neg = gen.new_full((batch, neg_count), float(weight_neg))

    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    target_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    dist = _cdist(old_gen, targets)
    weighted_dist = dist * target_weights[:, None, :]
    scale = weighted_dist.mean() / target_weights.mean().clamp_min(1e-6)
    scale_inputs = torch.clamp(scale / math.sqrt(max(feature_dim, 1)), min=1e-3)

    old_gen_scaled = old_gen / scale_inputs
    targets_scaled = targets / scale_inputs
    dist_normed = dist / torch.clamp(scale, min=1e-3)

    mask = torch.eye(gen_count, device=gen.device, dtype=gen.dtype)
    mask = F.pad(mask, (0, neg_count + pos_count)).unsqueeze(0)
    dist_normed = dist_normed + mask * 100.0

    force_across_r = torch.zeros_like(old_gen_scaled)
    split_idx = gen_count + neg_count

    for radius in r_list:
        logits = -dist_normed / radius
        affinity = torch.softmax(logits, dim=-1)
        affinity_t = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(affinity * affinity_t, min=1e-6))
        affinity = affinity * target_weights[:, None, :]

        aff_neg = affinity[:, :, :split_idx]
        aff_pos = affinity[:, :, split_idx:]
        sum_pos = aff_pos.sum(dim=-1, keepdim=True)
        sum_neg = aff_neg.sum(dim=-1, keepdim=True)

        coeff_neg = -aff_neg * sum_pos
        coeff_pos = aff_pos * sum_neg
        coeff = torch.cat([coeff_neg, coeff_pos], dim=-1)

        force = torch.einsum("biy,byx->bix", coeff, targets_scaled)
        force = force - coeff.sum(dim=-1, keepdim=True) * old_gen_scaled
        force_scale = torch.sqrt(torch.clamp((force ** 2).mean(), min=1e-8))
        force_across_r = force_across_r + force / force_scale

    goal = (old_gen_scaled + force_across_r).detach()
    gen_scaled = gen / scale_inputs
    return ((gen_scaled - goal) ** 2).mean()


class DriftFeatureLoss(nn.Module):
    def __init__(self, negative_weight=1.0, max_tokens=512, r_list=(0.02, 0.05, 0.2)):
        super().__init__()
        self.negative_weight = float(negative_weight)
        self.max_tokens = int(max_tokens)
        self.r_list = tuple(float(radius) for radius in r_list)

    def _sample_features(self, images, sample_count):
        batch = images.shape[0] // sample_count
        return [
            feature.reshape(batch, sample_count, -1)
            for feature in _feature_pyramid(images, self.max_tokens)
        ]

    def forward(self, candidates, positive_refs, negative_refs=None):
        batch, gen_count = candidates.shape[:2]
        pos_count = positive_refs.shape[1]

        gen_images = candidates.reshape(batch * gen_count, *candidates.shape[2:])
        pos_images = positive_refs.reshape(batch * pos_count, *positive_refs.shape[2:])
        gen_features = self._sample_features(gen_images, gen_count)
        pos_features = self._sample_features(pos_images, pos_count)

        neg_features = None
        neg_count = 0
        if negative_refs is not None and negative_refs.shape[1] > 0:
            neg_count = negative_refs.shape[1]
            neg_images = negative_refs.reshape(batch * neg_count, *negative_refs.shape[2:])
            neg_features = self._sample_features(neg_images, neg_count)

        total = candidates.new_tensor(0.0)
        for index, (gen_feature, pos_feature) in enumerate(zip(gen_features, pos_features)):
            neg_feature = None if neg_features is None else neg_features[index]
            weight_neg = None if neg_count == 0 else self.negative_weight
            total = total + drift_sample_loss(
                gen_feature,
                pos_feature,
                fixed_neg=neg_feature,
                weight_neg=weight_neg,
                r_list=self.r_list,
            )

        return total / len(gen_features)
