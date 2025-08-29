# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def compute_forgetting(acc_matrix: np.ndarray) -> float:
    T = acc_matrix.shape[0]
    F = []
    for j in range(T):
        prev = acc_matrix[:j, j] if j > 0 else np.array([0.0])
        if len(prev) == 0:
            F.append(0.0)
        else:
            F.append(max(prev) - acc_matrix[T-1, j])
    return float(np.mean(F))


def confusion_matrix(model, loader, num_classes: int, device: str) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            for t, p in zip(y.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy()):
                if t < num_classes and p < num_classes:
                    cm[t, p] += 1
    return cm


def save_confusion_matrix_pdf(cm: np.ndarray, out_pdf: str):
    sns.set_style('white')
    plt.figure(figsize=(4,3))
    norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(norm, cmap='Blues')
    plt.xlabel('pred'); plt.ylabel('true'); plt.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
