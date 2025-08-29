#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation helpers: accuracy, forgetting, and privacy metrics.
"""
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F


def evaluate_acc_propqem(backbone: torch.nn.Module, head: torch.nn.Module, loader, device: str) -> float:
    backbone.eval(); head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            f = backbone(x)
            logits = head(f)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def evaluate_acc_er(backbone: torch.nn.Module, head: torch.nn.Module, loader, device: str) -> float:
    return evaluate_acc_propqem(backbone, head, loader, device)


def average_forgetting(acc_matrix: np.ndarray) -> float:
    if acc_matrix.size == 0:
        return 0.0
    T = acc_matrix.shape[0]
    K = acc_matrix.shape[1]
    fgt = 0.0
    for k in range(K):
        max_past = np.max(acc_matrix[:T, k])
        final = acc_matrix[T-1, k]
        fgt += max(0.0, float(max_past - final))
    return fgt / max(1, K)


def privacy_auc_loss_threshold(model_head: torch.nn.Module,
                               member_feats: torch.Tensor,
                               non_member_feats: torch.Tensor,
                               labels: torch.Tensor,
                               device: str) -> float:
    model_head.eval()
    with torch.no_grad():
        logits_m = model_head(member_feats.to(device))
        logits_n = model_head(non_member_feats.to(device))
        y_m = labels[:logits_m.size(0)].to(device)
        y_n = labels[:logits_n.size(0)].to(device)
        loss_m = F.cross_entropy(logits_m, y_m, reduction='none').detach().cpu().numpy()
        loss_n = F.cross_entropy(logits_n, y_n, reduction='none').detach().cpu().numpy()
    scores = np.concatenate([loss_m, loss_n], axis=0)
    gt = np.concatenate([np.ones_like(loss_m), np.zeros_like(loss_n)], axis=0)
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
        auc = float(roc_auc_score(gt, -scores))
    except Exception:
        pos = scores[:len(loss_m)]; neg = scores[len(loss_m):]
        auc = float(np.mean(pos[:, None] < neg[None, :]))
    return auc
