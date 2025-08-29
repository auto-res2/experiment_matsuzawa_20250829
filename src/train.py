# -*- coding: utf-8 -*-
"""
Training module for FREQUENT and baseline ER.
Implements:
- Models (F0 backbone, adapter H, tail F1)
- Product Quantizer (pure PyTorch)
- Buffers (FrequentBuffer with ReLo pruning, ERBuffer)
- Trainers (FrequentTrainer, ERTrainer)
- Experiment runners (Exp1/Exp2/Exp3)

Notes
- Save all figures to .research/iteration1/images as high-quality PDFs.
- Use only relative imports inside src.
"""
from __future__ import annotations
import os
import math
import time
import copy
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .evaluate import (
    evaluate,
    confusion_matrix_torch,
    plot_curve,
    plot_accuracy_per_task,
    plot_confusion,
    plot_latency_hist,
)
from .preprocess import (
    PatternedDataset,
    build_task_splits,
    get_datasets,
)

# Optional energy logging (desktop GPU)
try:
    import pynvml
    _NVML_AVAILABLE = True
    try:
        pynvml.nvmlInit()
    except Exception:
        _NVML_AVAILABLE = False
except Exception:
    _NVML_AVAILABLE = False


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Backbones and Heads
# -----------------------------
class F0Backbone(nn.Module):
    """Early encoder F0 + d-bottleneck based on torchvision ResNet-18.
    Splits depth by taking first L residual layers.
    Output feature dimension = d via Linear after global avg pool.
    """
    def __init__(self, d: int = 128, freeze_depth: int = 2):
        super().__init__()
        try:
            import torchvision as tv
            # Handle both torchvision APIs
            try:
                base = tv.models.resnet18(weights=None)
            except TypeError:
                base = tv.models.resnet18(pretrained=False)
        except Exception as e:
            raise RuntimeError("torchvision is required for the backbone: pip install torchvision") from e
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self._feature_dim = 512
        self.freeze_depth = freeze_depth
        # Use up to layer{freeze_depth}
        self.used_layers = []
        if freeze_depth >= 0:
            self.used_layers += ['conv1', 'bn1', 'relu', 'maxpool']
        if freeze_depth >= 1: self.used_layers.append('layer1')
        if freeze_depth >= 2: self.used_layers.append('layer2')
        if freeze_depth >= 3: self.used_layers.append('layer3')
        if freeze_depth >= 4: self.used_layers.append('layer4')
        self.tap = nn.Linear(self._feature_dim, d)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        if 'layer1' in self.used_layers: x = self.layer1(x)
        if 'layer2' in self.used_layers: x = self.layer2(x)
        if 'layer3' in self.used_layers: x = self.layer3(x)
        if 'layer4' in self.used_layers: x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = self.tap(x)  # d-dim
        return z

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False


class AdapterH(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            eye = torch.eye(d)
            if self.proj.weight.shape == eye.shape:
                self.proj.weight.copy_(eye)

    def forward(self, x):
        return self.proj(x)


class TailF1(nn.Module):
    """Simple discriminative tail over d-dim features."""
    def __init__(self, d: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(d, num_classes)
        )
    def forward(self, z):
        return self.net(z)


# -----------------------------
# Product Quantizer (pure PyTorch)
# -----------------------------
class ProductQuantizerTorch:
    """M-subspace, K-centroids PQ with EMA centroid updates.
    - Codes stored as uint8 if K<=256; uint16 otherwise.
    - d must be divisible by M.
    Centroids are stored on CPU to minimize GPU VRAM.
    """
    def __init__(self, d: int, M: int = 8, K: int = 256, ema_alpha: float = 0.1, device: str = 'cpu'):
        assert d % M == 0, "d must be divisible by M"
        self.d = d
        self.M = M
        self.K = K
        self.d_sub = d // M
        self.ema_alpha = ema_alpha
        self.device = device
        # centroids: (M, K, d_sub)
        self.centroids = torch.zeros(M, K, self.d_sub, dtype=torch.float32, device=device)
        self.counts = torch.zeros(M, K, dtype=torch.long, device=device)
        self.initialized = False

    def _init_kmeans(self, Z: torch.Tensor, iters: int = 5):
        """Initialize centroids with a few iterations of k-means per subspace (mini-batch)."""
        assert Z.shape[1] == self.d
        N = Z.shape[0]
        Zs = Z.reshape(N, self.M, self.d_sub)
        with torch.no_grad():
            for m in range(self.M):
                X = Zs[:, m, :]
                idx = torch.randperm(N)[:self.K]
                C = X[idx].clone().contiguous()
                for _ in range(iters):
                    dist = torch.cdist(X, C, p=2)
                    a = dist.argmin(dim=1)
                    for k in range(self.K):
                        mask = (a == k)
                        if mask.any():
                            C[k] = X[mask].mean(dim=0)
                self.centroids[m] = C
                self.counts[m] = 1
        self.initialized = True

    @torch.no_grad()
    def encode(self, Z: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self._init_kmeans(Z)
        N = Z.shape[0]
        Zs = Z.reshape(N, self.M, self.d_sub)
        codes = []
        for m in range(self.M):
            X = Zs[:, m, :]
            C = self.centroids[m]
            dist = torch.cdist(X, C, p=2)
            a = dist.argmin(dim=1)
            codes.append(a)
        codes = torch.stack(codes, dim=1)  # (N, M)
        if self.K <= 256:
            return codes.to(torch.uint8).cpu()
        else:
            return codes.to(torch.int16).cpu()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
        device = device or self.centroids.device
        code_long = codes.to(torch.long).to(self.centroids.device)
        N = code_long.shape[0]
        parts = []
        for m in range(self.M):
            C = self.centroids[m]
            idx = code_long[:, m]
            part = C[idx]
            parts.append(part)
        Zrec = torch.cat(parts, dim=1)
        return Zrec.to(device)

    @torch.no_grad()
    def ema_update(self, Z: torch.Tensor, codes: torch.Tensor):
        alpha = self.ema_alpha
        code_long = codes.to(torch.long).to(self.centroids.device)
        N = Z.shape[0]
        Zs = Z.reshape(N, self.M, self.d_sub).to(self.centroids.device)
        for m in range(self.M):
            idx_m = code_long[:, m]
            for k in idx_m.unique():
                mask = (idx_m == k)
                if mask.any():
                    x_mean = Zs[mask, m, :].mean(dim=0)
                    self.centroids[m, k] = (1 - alpha) * self.centroids[m, k] + alpha * x_mean
                    self.counts[m, k] += mask.sum()

    def codebook_bytes(self) -> int:
        return self.centroids.numel() * 4  # float32


# -----------------------------
# Buffers: FREQUENT and ER
# -----------------------------
class FrequentBuffer:
    def __init__(self, pq: ProductQuantizerTorch, capacity_kb: int, label_bytes: int = 2, relo_T: int = 100):
        self.pq = pq
        self.capacity_kb = capacity_kb
        self.label_bytes = label_bytes
        self.codes: Optional[torch.Tensor] = None  # (N, M) uint8
        self.labels: Optional[torch.Tensor] = None  # (N,)
        self.relo = torch.tensor([])   # (N,)
        self.relo_counter = torch.tensor([], dtype=torch.int32)
        self.T = relo_T
        self.evictions = 0

    def per_sample_bytes(self) -> int:
        code_bytes = self.pq.M * (1 if self.pq.K <= 256 else 2)
        return code_bytes + self.label_bytes

    def fixed_overhead_bytes(self) -> int:
        return self.pq.codebook_bytes()

    def _current_bytes(self) -> int:
        n = 0 if self.codes is None else self.codes.shape[0]
        return self.fixed_overhead_bytes() + n * self.per_sample_bytes()

    def capacity_samples(self) -> int:
        usable = self.capacity_kb * 1024 - self.fixed_overhead_bytes()
        if usable <= 0:
            return 0
        return max(usable // self.per_sample_bytes(), 0)

    def size(self) -> int:
        return 0 if self.codes is None else self.codes.shape[0]

    def push(self, Z: torch.Tensor, y: torch.Tensor):
        codes = self.pq.encode(Z)
        self.pq.ema_update(Z, codes)
        if self.codes is None:
            self.codes = codes.clone()
            self.labels = y.detach().cpu().to(torch.int64)
            self.relo = torch.zeros(self.codes.shape[0], dtype=torch.float32)
            self.relo_counter = torch.zeros(self.codes.shape[0], dtype=torch.int32)
        else:
            self.codes = torch.cat([self.codes, codes.clone()], dim=0)
            self.labels = torch.cat([self.labels, y.detach().cpu().to(torch.int64)], dim=0)
            self.relo = torch.cat([self.relo, torch.zeros(codes.shape[0])], dim=0)
            self.relo_counter = torch.cat([self.relo_counter, torch.zeros(codes.shape[0], dtype=torch.int32)], dim=0)
        self._evict_if_needed()

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.codes is not None and self.labels is not None and self.size() > 0
        idx = torch.randint(0, self.size(), (min(n, self.size()),))
        codes = self.codes[idx]
        y = self.labels[idx]
        return codes, y, idx

    def decode(self, codes: torch.Tensor, device: str) -> torch.Tensor:
        return self.pq.decode(codes, device=device)

    def update_relo(self, idx: torch.Tensor, loss_cur: torch.Tensor, loss_tar: torch.Tensor):
        delta = (loss_cur.detach().cpu() - loss_tar.detach().cpu())
        self.relo[idx] = 0.9 * self.relo[idx] + 0.1 * delta
        decayed = self.relo[idx] <= 0
        self.relo_counter[idx] = torch.where(decayed, self.relo_counter[idx] + 1, torch.zeros_like(self.relo_counter[idx]))

    def _evict_if_needed(self):
        cap = self.capacity_samples()
        while self.size() > cap:
            mask_bad = (self.relo <= 0) & (self.relo_counter >= self.T)
            bad_idx = mask_bad.nonzero(as_tuple=False).flatten()
            if bad_idx.numel() == 0:
                kick = torch.tensor([0])
            else:
                kick = bad_idx[:1]
            self._remove_indices(kick)
            self.evictions += int(kick.numel())

    def _remove_indices(self, idx: torch.Tensor):
        keep = torch.ones(self.size(), dtype=torch.bool)
        keep[idx] = False
        self.codes = self.codes[keep]
        self.labels = self.labels[keep]
        self.relo = self.relo[keep]
        self.relo_counter = self.relo_counter[keep]


class ERBuffer:
    """Stores raw images (uint8) and labels."""
    def __init__(self, capacity_kb: int, img_shape=(3, 32, 32), label_bytes: int = 2):
        self.capacity_kb = capacity_kb
        self.img_shape = img_shape
        self.label_bytes = label_bytes
        self.images: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None

    def per_sample_bytes(self) -> int:
        C, H, W = self.img_shape
        return C * H * W + self.label_bytes

    def fixed_overhead_bytes(self) -> int:
        return 0

    def size(self) -> int:
        return 0 if self.images is None else self.images.shape[0]

    def capacity_samples(self) -> int:
        usable = self.capacity_kb * 1024 - self.fixed_overhead_bytes()
        return max(usable // self.per_sample_bytes(), 0)

    def push(self, x: torch.Tensor, y: torch.Tensor):
        x8 = (x.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
        if self.images is None:
            self.images = x8.clone()
            self.labels = y.detach().cpu().to(torch.int64)
        else:
            self.images = torch.cat([self.images, x8.clone()], dim=0)
            self.labels = torch.cat([self.labels, y.detach().cpu().to(torch.int64)], dim=0)
        cap = self.capacity_samples()
        if self.size() > cap and cap > 0:
            overflow = self.size() - cap
            self.images = self.images[overflow:]
            self.labels = self.labels[overflow:]

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.images is not None and self.labels is not None and self.size() > 0
        idx = torch.randint(0, self.size(), (min(n, self.size()),))
        x = self.images[idx].to(torch.float32) / 255.0
        y = self.labels[idx]
        return x, y, idx


# -----------------------------
# Training/Evaluation helpers
# -----------------------------
@dataclass
class TrainConfig:
    device: str = 'auto'  # 'auto' -> cuda if available
    d: int = 128
    M: int = 8
    K: int = 256
    freeze_depth: int = 2
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs_per_task: int = 1
    patience: int = 1
    replay_ratio_target_budget: float = 1.0
    seed_list: List[int] = field(default_factory=lambda: [11])
    num_workers: int = 0
    mem_kb_list: List[int] = field(default_factory=lambda: [200])
    n_tasks: int = 2
    classes_per_task: int = 5
    use_fake_data: bool = True
    dataset_name: str = 'CIFAR100'
    subset_per_class: int = 100
    data_patterns: List[str] = field(default_factory=lambda: ['standard'])


def choose_replay_ratio(target_budget: float, f_online: float = 1.0, f_replay: float = 0.65) -> float:
    if f_online <= target_budget:
        denom = (f_replay - target_budget)
        if denom <= 0:
            return 1.0
        num = (target_budget - f_online)
        r = max(num / denom, 0.0)
        return float(max(r, 0.0))
    else:
        return 0.0


# -----------------------------
# FREQUENT trainer and Baseline ER trainer
# -----------------------------
class FrequentTrainer:
    def __init__(self, cfg: TrainConfig, num_classes: int, memory_kb: int,
                 relo_T: int = 50, refresh_every_tasks: Optional[int] = None):
        self.cfg = cfg
        self.device = ("cuda" if (cfg.device == 'auto' and torch.cuda.is_available()) else (cfg.device if cfg.device != 'auto' else 'cpu'))
        self.num_classes = num_classes
        self.memory_kb = memory_kb
        # Models
        self.F0 = F0Backbone(d=cfg.d, freeze_depth=cfg.freeze_depth).to(self.device)
        self.H = AdapterH(d=cfg.d).to(self.device)
        self.F1 = TailF1(d=cfg.d, num_classes=num_classes).to(self.device)
        # EMA target networks
        self.H_ema = copy.deepcopy(self.H).to(self.device)
        self.F1_ema = copy.deepcopy(self.F1).to(self.device)
        for p in self.H_ema.parameters(): p.requires_grad = False
        for p in self.F1_ema.parameters(): p.requires_grad = False
        # PQ and buffer
        self.pq = ProductQuantizerTorch(d=cfg.d, M=cfg.M, K=cfg.K, ema_alpha=0.1, device='cpu')
        self.buffer = FrequentBuffer(self.pq, capacity_kb=memory_kb, relo_T=relo_T)
        # Optimizer (F0 remains frozen during these updates)
        self.optimizer = torch.optim.SGD(list(self.H.parameters()) + list(self.F1.parameters()),
                                         lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.refresh_every_tasks = refresh_every_tasks
        # Replay ratio under target compute budget
        self.r_replay = choose_replay_ratio(cfg.replay_ratio_target_budget, f_online=1.0, f_replay=0.65)
        # Logs
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.task_acc_history: List[List[float]] = []
        self.max_acc_per_task: Dict[int, float] = {}

    def _update_ema(self, decay: float = 0.99):
        with torch.no_grad():
            for p, p_ema in zip(self.H.parameters(), self.H_ema.parameters()):
                p_ema.copy_(decay*p_ema + (1-decay)*p)
            for p, p_ema in zip(self.F1.parameters(), self.F1_ema.parameters()):
                p_ema.copy_(decay*p_ema + (1-decay)*p)

    def _codebook_refresh(self, loader_small: DataLoader, steps: int = 50):
        self.F0.eval()
        with torch.no_grad():
            cnt = 0
            for x, _ in loader_small:
                x = x.to(self.device)
                z = self.F0(x)
                codes = self.pq.encode(z)
                self.pq.ema_update(z, codes)
                cnt += 1
                if cnt >= steps:
                    break
        print(f"[Refresh] Codebook refreshed with {cnt} mini-batches.")

    def train_task(self, task_id: int, train_loader: DataLoader, val_loader: DataLoader):
        print(f"\n[Task {task_id}] Starting training | Memory budget: {self.memory_kb} kB | r_replay={self.r_replay:.2f}")
        print(f"[Task {task_id}] Codebook bytes={self.buffer.fixed_overhead_bytes()} | per-sample bytes={self.buffer.per_sample_bytes()} | capacity samples={self.buffer.capacity_samples()}")
        best_val = float('inf')
        bad_epochs = 0
        for epoch in range(self.cfg.epochs_per_task):
            self.F0.eval()
            self.H.train(); self.F1.train()
            epoch_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                # Online forward
                with torch.no_grad():
                    z = self.F0(xb)
                logits_online = self.F1(self.H(z))
                loss_online = self.criterion(logits_online, yb)
                loss = loss_online
                # Replay (decoder-free)
                if self.buffer.size() > 0 and random.random() < (self.r_replay/(1+self.r_replay)):
                    codes, y_mem, idx_mem = self.buffer.sample(self.cfg.batch_size)
                    z_mem = self.buffer.decode(codes, device=self.device)
                    logits_mem = self.F1(self.H(z_mem))
                    loss_mem = self.criterion(logits_mem, y_mem.to(self.device))
                    loss = loss + loss_mem
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.train_losses.append(float(loss.item()))
                epoch_losses.append(float(loss.item()))
                # EMA update
                self._update_ema(decay=0.99)
                # ReLo update
                if self.buffer.size() > 0 and 'loss_mem' in locals():
                    with torch.no_grad():
                        per_sample_cur = F.cross_entropy(logits_mem, y_mem.to(self.device), reduction='none')
                        logits_tar = self.F1_ema(self.H_ema(z_mem))
                        per_sample_tar = F.cross_entropy(logits_tar, y_mem.to(self.device), reduction='none')
                        self.buffer.update_relo(idx_mem, per_sample_cur, per_sample_tar)
                # Push to buffer
                with torch.no_grad():
                    self.buffer.push(z, yb)
            acc_val, val_loss, _, _ = evaluate(self.F0, self.H, self.F1, val_loader, device=self.device)
            self.val_losses.append(float(val_loss))
            print(f"[Task {task_id}] Epoch {epoch+1}/{self.cfg.epochs_per_task} | TrainLoss={np.mean(epoch_losses):.4f} | ValLoss={val_loss:.4f} | ValAcc={acc_val*100:.2f}% | BufferSize={self.buffer.size()} | Evictions={self.buffer.evictions}")
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= self.cfg.patience:
                print(f"[Task {task_id}] Validation plateau reached. F0 remains frozen for decoder-free replay.")
                bad_epochs = 0
        # Optional codebook refresh across tasks
        if self.refresh_every_tasks is not None and (task_id + 1) % self.refresh_every_tasks == 0:
            small_loader = DataLoader(train_loader.dataset, batch_size=64, shuffle=True, num_workers=0)
            self._codebook_refresh(small_loader, steps=10)

    def evaluate_upto_task(self, test_loaders: List[DataLoader]) -> Tuple[List[float], float, float]:
        accs = []
        for i, loader in enumerate(test_loaders):
            acc, _, y_true, y_pred = evaluate(self.F0, self.H, self.F1, loader, self.cfg.device if self.cfg.device != 'auto' else self.device)
            accs.append(acc)
            self.max_acc_per_task[i] = max(self.max_acc_per_task.get(i, 0.0), acc)
        avg_acc = float(np.mean(accs)) if len(accs) > 0 else 0.0
        forgetting = []
        for i, a in enumerate(accs):
            max_prev = self.max_acc_per_task.get(i, a)
            forgetting.append(max(0.0, float(max_prev - a)))
        avg_forgetting = float(np.mean(forgetting)) if len(forgetting) > 0 else 0.0
        return accs, avg_acc, avg_forgetting


class ERTrainer:
    def __init__(self, cfg: TrainConfig, num_classes: int, memory_kb: int):
        self.cfg = cfg
        self.device = ("cuda" if (cfg.device == 'auto' and torch.cuda.is_available()) else (cfg.device if cfg.device != 'auto' else 'cpu'))
        self.num_classes = num_classes
        self.memory_kb = memory_kb
        self.F0 = F0Backbone(d=cfg.d, freeze_depth=cfg.freeze_depth).to(self.device)
        self.H = AdapterH(d=cfg.d).to(self.device)
        self.F1 = TailF1(d=cfg.d, num_classes=num_classes).to(self.device)
        params = list(self.F0.parameters()) + list(self.H.parameters()) + list(self.F1.parameters())
        self.optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.buffer = ERBuffer(capacity_kb=memory_kb, img_shape=(3, 32, 32))
        self.r_replay = choose_replay_ratio(cfg.replay_ratio_target_budget, f_online=1.0, f_replay=1.0)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.task_acc_history: List[List[float]] = []
        self.max_acc_per_task: Dict[int, float] = {}

    def train_task(self, task_id: int, train_loader: DataLoader, val_loader: DataLoader):
        print(f"\n[ER Task {task_id}] Starting | Memory {self.memory_kb} kB | r_replay={self.r_replay:.2f}")
        print(f"[ER Task {task_id}] Per-sample bytes={self.buffer.per_sample_bytes()} | capacity samples={self.buffer.capacity_samples()}")
        best_val = float('inf')
        bad_epochs = 0
        for epoch in range(self.cfg.epochs_per_task):
            self.F0.train(); self.H.train(); self.F1.train()
            epoch_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                z = self.F0(xb)
                logits = self.F1(self.H(z))
                loss = self.criterion(logits, yb)
                if self.buffer.size() > 0 and random.random() < (self.r_replay/(1+self.r_replay)):
                    x_mem, y_mem, _ = self.buffer.sample(self.cfg.batch_size)
                    x_mem = x_mem.to(self.device); y_mem = y_mem.to(self.device)
                    z_mem = self.F0(x_mem)
                    logits_mem = self.F1(self.H(z_mem))
                    loss = loss + self.criterion(logits_mem, y_mem)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.train_losses.append(float(loss.item()))
                epoch_losses.append(float(loss.item()))
                self.buffer.push(xb.cpu(), yb.cpu())
            acc_val, val_loss, _, _ = evaluate(self.F0, self.H, self.F1, val_loader, device=self.device)
            self.val_losses.append(float(val_loss))
            print(f"[ER Task {task_id}] Epoch {epoch+1}/{self.cfg.epochs_per_task} | TrainLoss={np.mean(epoch_losses):.4f} | ValLoss={val_loss:.4f} | ValAcc={acc_val*100:.2f}% | BufferSize={self.buffer.size()}")
            if val_loss < best_val - 1e-4:
                best_val = val_loss; bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= self.cfg.patience:
                print(f"[ER Task {task_id}] Validation plateau observed.")
                bad_epochs = 0

    def evaluate_upto_task(self, test_loaders: List[DataLoader]) -> Tuple[List[float], float, float]:
        accs = []
        for i, loader in enumerate(test_loaders):
            acc, _, y_true, y_pred = evaluate(self.F0, self.H, self.F1, loader, self.device)
            accs.append(acc)
            self.max_acc_per_task[i] = max(self.max_acc_per_task.get(i, 0.0), acc)
        avg_acc = float(np.mean(accs)) if len(accs) > 0 else 0.0
        forgetting = []
        for i, a in enumerate(accs):
            max_prev = self.max_acc_per_task.get(i, a)
            forgetting.append(max(0.0, float(max_prev - a)))
        avg_forgetting = float(np.mean(forgetting)) if len(forgetting) > 0 else 0.0
        return accs, avg_acc, avg_forgetting


# -----------------------------
# Experiment 1 – strict memory & compute budgets (FREQUENT vs ER)
# -----------------------------

def run_experiment1_strict(cfg: TrainConfig, img_dir: str):
    print("\n=====================\nExperiment 1: Strict memory & compute budgets\n=====================")
    for pattern in cfg.data_patterns:
        print(f"\n[Exp1] Data pattern: {pattern}")
        train_tasks, test_tasks, num_classes = get_datasets(cfg, pattern=pattern)
        task_train_loaders = [DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers) for ds in train_tasks]
        task_val_loaders = [DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers) for ds in train_tasks]
        task_test_loaders = [DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers) for ds in test_tasks]
        for mem_kb in cfg.mem_kb_list:
            print(f"\n[Exp1] Memory budget: {mem_kb} kB | Compute budget factor: {cfg.replay_ratio_target_budget}")
            fq = FrequentTrainer(cfg, num_classes=num_classes, memory_kb=mem_kb, relo_T=50, refresh_every_tasks=2)
            er = ERTrainer(cfg, num_classes=num_classes, memory_kb=mem_kb)
            fq_acc_hist, er_acc_hist = [], []
            for t in range(cfg.n_tasks):
                fq.train_task(t, task_train_loaders[t], task_val_loaders[t])
                accs_fq, avg_acc_fq, avg_forget_fq = fq.evaluate_upto_task(task_test_loaders[:t+1])
                fq_acc_hist.append(accs_fq)
                print(f"[FREQUENT] After task {t}: AvgAcc={avg_acc_fq:.3f} | AvgForgetting={avg_forget_fq:.3f}")

                er.train_task(t, task_train_loaders[t], task_val_loaders[t])
                accs_er, avg_acc_er, avg_forget_er = er.evaluate_upto_task(task_test_loaders[:t+1])
                er_acc_hist.append(accs_er)
                print(f"[ER] After task {t}: AvgAcc={avg_acc_er:.3f} | AvgForgetting={avg_forget_er:.3f}")

            # Figures
            plot_curve(list(range(len(fq.train_losses))), fq.train_losses,
                       title='Training Loss (FREQUENT)', ylabel='Loss',
                       filename=os.path.join(img_dir, f'training_loss_frequent_{pattern}_{mem_kb}kB.pdf'))
            plot_curve(list(range(len(er.train_losses))), er.train_losses,
                       title='Training Loss (ER)', ylabel='Loss',
                       filename=os.path.join(img_dir, f'training_loss_er_{pattern}_{mem_kb}kB.pdf'))
            plot_accuracy_per_task(fq_acc_hist, 'FREQUENT', filename=os.path.join(img_dir, f'accuracy_frequent_{pattern}_{mem_kb}kB.pdf'))
            plot_accuracy_per_task(er_acc_hist, 'ER', filename=os.path.join(img_dir, f'accuracy_er_{pattern}_{mem_kb}kB.pdf'))
            # Confusions
            from torch.utils.data import DataLoader as _DL
            acc_fq, _, y_true_fq, y_pred_fq = evaluate(fq.F0, fq.H, fq.F1, task_test_loaders[-1], device=(cfg.device if cfg.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu')))
            acc_er, _, y_true_er, y_pred_er = evaluate(er.F0, er.H, er.F1, task_test_loaders[-1], device=(cfg.device if cfg.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu')))
            cm_fq = confusion_matrix_torch(y_true_fq, y_pred_fq, num_classes)
            cm_er = confusion_matrix_torch(y_true_er, y_pred_er, num_classes)
            plot_confusion(cm_fq, filename=os.path.join(img_dir, f'confusion_frequent_{pattern}_{mem_kb}kB.pdf'))
            plot_confusion(cm_er, filename=os.path.join(img_dir, f'confusion_er_{pattern}_{mem_kb}kB.pdf'))

            print(f"[Exp1][{pattern}] Memory={mem_kb} kB | FREQUENT final avg acc across tasks: {np.mean([np.mean(a) for a in fq_acc_hist]):.3f}")
            print(f"[Exp1][{pattern}] Memory={mem_kb} kB | ER final avg acc across tasks: {np.mean([np.mean(a) for a in er_acc_hist]):.3f}")
            print(f"[Exp1][{pattern}] FREQUENT buffer capacity (samples): {fq.buffer.capacity_samples()} | codebook bytes: {fq.buffer.fixed_overhead_bytes()}")
            print(f"[Exp1][{pattern}] ER buffer capacity (samples): {er.buffer.capacity_samples()} | per-sample bytes: {er.buffer.per_sample_bytes()}")


# -----------------------------
# Experiment 2 – Ablations & drift robustness
# -----------------------------

def run_experiment2_ablation(cfg: TrainConfig, img_dir: str):
    print("\n=====================\nExperiment 2: Ablations & Drift Robustness\n=====================")
    mem_kb = cfg.mem_kb_list[0]
    freeze_depth_list = [max(0, cfg.freeze_depth-1), cfg.freeze_depth]
    bytes_per_sample_settings = [(cfg.M, cfg.K), (max(2, cfg.M//2), 256)]
    relo_on_list = [True, False]
    refresh_list = [True, False]
    results = []
    pattern = 'tinted'
    train_tasks, test_tasks, num_classes = get_datasets(cfg, pattern=pattern)
    task_train_loaders = [DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers) for ds in train_tasks]
    task_val_loaders = [DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers) for ds in train_tasks]
    task_test_loaders = [DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers) for ds in test_tasks]

    for L in freeze_depth_list:
        for (M, K) in bytes_per_sample_settings:
            d = cfg.d
            for relo_on in relo_on_list:
                for refresh in refresh_list:
                    cfg2 = copy.deepcopy(cfg)
                    cfg2.freeze_depth = L
                    cfg2.M = M; cfg2.K = K; cfg2.d = d
                    fq = FrequentTrainer(cfg2, num_classes=num_classes, memory_kb=mem_kb,
                                         relo_T=(50 if relo_on else 10),
                                         refresh_every_tasks=(2 if refresh else None))
                    fq_acc_hist = []
                    for t in range(cfg2.n_tasks):
                        fq.train_task(t, task_train_loaders[t], task_val_loaders[t])
                        accs_fq, avg_acc_fq, avg_forget_fq = fq.evaluate_upto_task(task_test_loaders[:t+1])
                        fq_acc_hist.append(accs_fq)
                        print(f"[Ablation] L={L} M={M} ReLo={'on' if relo_on else 'off'} Refresh={'on' if refresh else 'off'} | Task {t} AvgAcc={avg_acc_fq:.3f} Forget={avg_forget_fq:.3f}")
                    final_avg = float(np.mean([np.mean(a) for a in fq_acc_hist]))
                    results.append({
                        'freeze_depth': L,
                        'M': M,
                        'relo': relo_on,
                        'refresh': refresh,
                        'final_avg_acc': final_avg,
                        'buffer_capacity_samples': fq.buffer.capacity_samples(),
                        'codebook_bytes': fq.buffer.fixed_overhead_bytes(),
                    })
    # Bar chart of final_avg_acc per setting
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        labels = [f"L{r['freeze_depth']}_M{r['M']}_ReLo{'1' if r['relo'] else '0'}_Ref{'1' if r['refresh'] else '0'}" for r in results]
        vals = [r['final_avg_acc'] for r in results]
        plt.figure(figsize=(max(6, len(labels)*0.5), 3))
        sns.barplot(x=labels, y=vals, color='steelblue')
        plt.ylabel('Final Avg Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.title('Ablation study — FREQUENT')
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'accuracy_ablation.pdf'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[Exp2] Plotting failed: {e}")

    print("[Exp2] Ablation results:")
    for r in results:
        print(r)

    # Drift robustness: induce extra training on current task w/o replay
    print("\n[Exp2] Drift robustness test (no replay for 3 extra epochs at mid stream)")
    cfg3 = copy.deepcopy(cfg)
    train_tasks, test_tasks, num_classes = get_datasets(cfg3, pattern='standard')
    task_train_loaders = [DataLoader(ds, batch_size=cfg3.batch_size, shuffle=True, num_workers=cfg3.num_workers) for ds in train_tasks]
    task_test_loaders = [DataLoader(ds, batch_size=cfg3.batch_size, shuffle=False, num_workers=cfg3.num_workers) for ds in test_tasks]
    mem_kb = cfg3.mem_kb_list[0]
    fq_no_refresh = FrequentTrainer(cfg3, num_classes=num_classes, memory_kb=mem_kb, refresh_every_tasks=None)
    fq_refresh = FrequentTrainer(cfg3, num_classes=num_classes, memory_kb=mem_kb, refresh_every_tasks=2)

    mid = max(1, cfg3.n_tasks//2)
    for t in range(mid):
        fq_no_refresh.train_task(t, task_train_loaders[t], task_train_loaders[t])
        fq_refresh.train_task(t, task_train_loaders[t], task_train_loaders[t])

    def induce_drift(tr: FrequentTrainer, loader: DataLoader, extra_epochs: int = 3):
        tr.F0.eval(); tr.H.train(); tr.F1.train()
        for _ in range(extra_epochs):
            for xb, yb in loader:
                xb = xb.to(tr.device); yb = yb.to(tr.device)
                with torch.no_grad():
                    z = tr.F0(xb)
                logits = tr.F1(tr.H(z))
                loss = F.cross_entropy(logits, yb)
                tr.optimizer.zero_grad(); loss.backward(); tr.optimizer.step()

    induce_drift(fq_no_refresh, task_train_loaders[mid-1], extra_epochs=3)
    induce_drift(fq_refresh, task_train_loaders[mid-1], extra_epochs=3)

    for t in range(mid, cfg3.n_tasks):
        fq_no_refresh.train_task(t, task_train_loaders[t], task_train_loaders[t])
        fq_refresh.train_task(t, task_train_loaders[t], task_train_loaders[t])

    accs_no, avg_no, _ = fq_no_refresh.evaluate_upto_task(task_test_loaders)
    accs_rf, avg_rf, _ = fq_refresh.evaluate_upto_task(task_test_loaders)
    print(f"[Exp2 Drift] Final avg acc without refresh: {avg_no:.3f} | with refresh: {avg_rf:.3f}")


# -----------------------------
# Experiment 3 – Stream, latency, and optional energy logging
# -----------------------------

def run_experiment3_stream(cfg: TrainConfig, img_dir: str):
    print("\n=====================\nExperiment 3: Stream & Latency/Energy\n=====================")
    pattern = 'sketch'
    train_tasks, test_tasks, num_classes = get_datasets(cfg, pattern=pattern)
    from torch.utils.data import ConcatDataset
    from torch.utils.data import DataLoader as _DL
    stream_dataset = ConcatDataset(train_tasks)
    stream_loader = _DL(stream_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = _DL(ConcatDataset(test_tasks), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    mem_kb = cfg.mem_kb_list[0]
    fq = FrequentTrainer(cfg, num_classes=num_classes, memory_kb=mem_kb, relo_T=50, refresh_every_tasks=None)

    latencies = []
    energies = []
    steps = 0

    if _NVML_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    def read_power_w():
        if not _NVML_AVAILABLE:
            return 0.0
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

    fq.F0.eval()
    fq.H.train(); fq.F1.train()
    tmax = 50  # quick
    for xb, yb in stream_loader:
        t0 = time.time()
        xb = xb.to(fq.device); yb = yb.to(fq.device)
        with torch.no_grad():
            z = fq.F0(xb)
        logits = fq.F1(fq.H(z))
        loss = F.cross_entropy(logits, yb)
        fq.optimizer.zero_grad(); loss.backward(); fq.optimizer.step()
        with torch.no_grad():
            fq.buffer.push(z, yb)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)
        energies.append(read_power_w() * (t1 - t0))
        steps += 1
        if steps % 10 == 0:
            print(f"[Exp3] Step {steps} | Loss={loss.item():.4f} | Latency={latencies[-1]:.2f} ms | Buffer={fq.buffer.size()}")
        if steps >= tmax:
            break

    acc, loss_t, _, _ = evaluate(fq.F0, fq.H, fq.F1, test_loader, device=fq.device)
    print(f"[Exp3] Stream eval — Acc={acc:.3f} | TestLoss={loss_t:.3f} | AvgLatency={np.mean(latencies):.2f} ms | Energy/step (if NVML)={np.mean(energies):.4f} J")

    plot_latency_hist(latencies, filename=os.path.join(img_dir, 'inference_latency_frequent.pdf'))
