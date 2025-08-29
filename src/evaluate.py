# -*- coding: utf-8 -*-
"""
Evaluation and plotting utilities for FREQUENT experiments.
- evaluate(): forward pass and metrics
- confusion_matrix_torch(): simple numpy confusion matrix
- Plot helpers: curve, accuracy per task, confusion heatmap, latency histogram
All plots saved as high-quality PDFs suitable for academic papers.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, List


def evaluate(model_f0: nn.Module, H: nn.Module, F1: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model_f0.eval(); H.eval(); F1.eval()
    total = 0
    correct = 0
    y_true_all = []
    y_pred_all = []
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            z = model_f0(x)
            logits = F1(H(z))
            loss = criterion(logits, y)
            loss_sum += float(loss.item())
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += int((pred == y).sum().item())
            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())
    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    return acc, avg_loss, y_true_all, y_pred_all


def confusion_matrix_torch(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# -----------------------------
# Plotting helpers
# -----------------------------

def plot_curve(xs: List[float], ys: List[float], title: str, ylabel: str, filename: str):
    plt.figure(figsize=(5, 3))
    plt.plot(xs, ys, lw=2)
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_accuracy_per_task(acc_hist: List[List[float]], method_name: str, filename: str):
    plt.figure(figsize=(5, 3))
    means = [np.mean(a) for a in acc_hist] if len(acc_hist)>0 else []
    plt.plot(range(1, len(means)+1), means, marker='o', lw=2)
    plt.title(f'Average accuracy across seen tasks — {method_name}')
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_confusion(cm: np.ndarray, filename: str):
    plt.figure(figsize=(4.5, 4.0))
    sns.heatmap(cm, cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_latency_hist(latencies_ms: List[float], filename: str):
    plt.figure(figsize=(5, 3))
    sns.histplot(latencies_ms, bins=15, kde=True)
    plt.xlabel('Latency per update (ms)')
    plt.title('Update latency distribution — FREQUENT stream')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
