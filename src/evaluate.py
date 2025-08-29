"""
Evaluation utilities for ProPqEM experiments.
"""
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [to_device(xx, device) for xx in x]
    return x.to(device)


def compute_confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    return cm


def evaluate(model_f, model_cls, loader, device: torch.device, num_classes: int) -> Tuple[float, np.ndarray, np.ndarray]:
    model_f.eval(); model_cls.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for x, y in loader:
            x = to_device(x, device)
            y = to_device(y, device)
            f = model_f(x)
            logits = model_cls(f)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().numpy())
            gts.append(y.cpu().numpy())
    preds = np.concatenate(preds) if len(preds) else np.array([])
    gts = np.concatenate(gts) if len(gts) else np.array([])
    acc = float((preds == gts).mean()) if len(preds) else 0.0
    return acc, preds, gts


def compute_forgetting(acc_matrix):
    # acc_matrix[t][j]: accuracy on task j after training task t
    T = len(acc_matrix)
    fgt = []
    for j in range(T):
        best = max(acc_matrix[t][j] for t in range(j, T)) if j < T else 0.0
        last = acc_matrix[-1][j] if j < len(acc_matrix[-1]) else 0.0
        fgt.append(max(0.0, best - last))
    return fgt


def roc_auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    # Scores: higher = more member-like; Labels: 1=member,0=non-member
    order = np.argsort(scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return 0.5
    tps = 0
    fps = 0
    prev_score = -np.inf
    points = []
    for i in range(len(scores_sorted)):
        s = scores_sorted[i]
        if s != prev_score:
            tpr = tps / P if P > 0 else 0.0
            fpr = fps / N if N > 0 else 0.0
            points.append((fpr, tpr))
            prev_score = s
        if labels_sorted[i] == 1:
            tps += 1
        else:
            fps += 1
    tpr = tps / P
    fpr = fps / N
    points.append((fpr, tpr))
    points = sorted(points, key=lambda x: x[0])
    auc = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return float(max(0.0, min(1.0, auc)))
