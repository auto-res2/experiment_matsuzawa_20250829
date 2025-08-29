"""
Training utilities and core ProPqEM implementation.
Run from project root with: python -m src.main
All plots saved as high-quality PDFs under .research/iteration1/images by default.
"""
import os
import math
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .evaluate import evaluate, compute_confusion_matrix, compute_forgetting

# ------------------------------
# Utilities & Reproducibility
# ------------------------------

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [to_device(xx, device) for xx in x]
    return x.to(device)


def count_bytes_tensor(t: torch.Tensor) -> int:
    if t.dtype == torch.uint8:
        return t.numel()
    elif t.dtype == torch.float32:
        return t.numel() * 4
    elif t.dtype == torch.float16:
        return t.numel() * 2
    elif t.dtype == torch.int64:
        return t.numel() * 8
    else:
        return t.element_size() * t.numel()


def bytes_to_mb(nbytes: int) -> float:
    return nbytes / (1024.0 * 1024.0)


def running_mean(prev_mean, new_val, count):
    return (prev_mean * (count - 1) + new_val) / max(count, 1)


def cosine_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    cos = (a_n * b_n).sum(dim=-1)
    return 1.0 - cos


# ------------------------------
# Model Components
# ------------------------------

class TinyConvBackbone(nn.Module):
    """
    Lightweight CNN backbone producing a 256-D feature f.
    Split into 4 blocks; we will freeze bottom-3 blocks after warm-up.
    """
    def __init__(self, out_dim=256):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(256, out_dim)

        self.blocks = [self.block1, self.block2, self.block3, self.block4]
        self.frozen_until = 2  # freeze blocks 0..2 after warm-up

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        f = self.proj(x)
        return f

    def freeze_low_blocks(self):
        for bi, blk in enumerate(self.blocks[:self.frozen_until+1]):
            for p in blk.parameters():
                p.requires_grad = False


class ResidualMLP(nn.Module):
    # 256 -> 128 -> 64 as task-adaptive residual encoder
    def __init__(self, in_dim=256, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, f):
        return self.net(f)


class DeltaDecoder(nn.Module):
    # z (64) -> f (256)
    def __init__(self, in_dim=64, hidden=128, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, z):
        return self.net(z)


class Classifier(nn.Module):
    def __init__(self, in_dim=256, num_classes=100):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, f):
        return self.fc(f)


# ------------------------------
# Product Quantiser with Generations
# ------------------------------

class ProductQuantiser(nn.Module):
    """
    PQ with M subspaces, Ks codes per subspace, dimension D=64.
    Manages generations. Each generation contains M codebooks [Ks, ds].
    """
    def __init__(self, D=64, M=8, Ks=256, device='cpu'):
        super().__init__()
        assert D % M == 0
        self.D = D
        self.M = M
        self.Ks = Ks
        self.ds = D // M
        self.generations: List[List[torch.Tensor]] = []
        self.device = device
        self.add_new_generation(init_means=None)

    def add_new_generation(self, init_means: Optional[torch.Tensor]):
        gen = []
        if init_means is None:
            for m in range(self.M):
                cb = torch.randn(self.Ks, self.ds, device=self.device)
                cb = F.normalize(cb, dim=-1)
                gen.append(nn.Parameter(cb))
        else:
            with torch.no_grad():
                init_means = init_means.to(self.device)
                init_means = init_means.view(-1, self.M, self.ds)
                for m in range(self.M):
                    sub = init_means[:, m]
                    idx = torch.randint(0, sub.size(0), (1,), device=self.device)
                    centers = [sub[idx].squeeze(0)]
                    for _ in range(self.Ks - 1):
                        d2 = torch.cdist(sub, torch.stack(centers)) ** 2
                        probs = d2.min(dim=1).values + 1e-6
                        probs = probs / probs.sum()
                        idx = torch.multinomial(probs, 1)
                        centers.append(sub[idx].squeeze(0))
                    cb = torch.stack(centers)
                    cb = F.normalize(cb, dim=-1)
                    gen.append(nn.Parameter(cb))
        for i, cb in enumerate(gen):
            self.register_parameter(f'gen{len(self.generations)}_cb{i}', cb)
        self.generations.append(gen)

    def current_generation_id(self) -> int:
        return len(self.generations) - 1

    def generation_count(self) -> int:
        return len(self.generations)

    def encode(self, z: torch.Tensor) -> torch.ByteTensor:
        gen = self.generations[-1]
        B = z.size(0)
        z_parts = z.view(B, self.M, self.ds)
        codes = []
        for m in range(self.M):
            cb = gen[m]  # [Ks, ds]
            d = torch.cdist(z_parts[:, m], cb)
            idx = torch.argmin(d, dim=1)
            codes.append(idx.to(torch.uint8))
        codes = torch.stack(codes, dim=1)
        return codes

    def decode(self, codes: torch.ByteTensor, gen_id: int) -> torch.Tensor:
        gen = self.generations[gen_id]
        B = codes.size(0)
        parts = []
        for m in range(self.M):
            idx = codes[:, m].long()
            parts.append(gen[m][idx])
        z_rec = torch.cat(parts, dim=1)
        return z_rec

    def decode_mixed_gens(self, codes: torch.ByteTensor, gens: torch.LongTensor) -> torch.Tensor:
        # Handles batch where each item can come from a different generation
        if codes.numel() == 0:
            return torch.empty(0, self.D, device=codes.device)
        outs = torch.empty(codes.size(0), self.D, device=codes.device)
        unique_gens = gens.unique()
        for g in unique_gens:
            mask = (gens == g)
            idxs = mask.nonzero(as_tuple=False).view(-1)
            z_g = self.decode(codes[idxs], int(g.item()))
            outs[idxs] = z_g
        return outs

    def recon_error(self, z: torch.Tensor) -> torch.Tensor:
        codes = self.encode(z)
        z_rec = self.decode(codes, self.current_generation_id())
        return cosine_error(z, z_rec).mean()

    def bytes_for_codebooks(self) -> int:
        total = 0
        for gen in self.generations:
            for cb in gen:
                total += count_bytes_tensor(cb.data)
        return total


# ------------------------------
# GIS Selector
# ------------------------------

class GISState:
    def __init__(self, num_classes: int, rho: float = 1.0):
        self.num_classes = num_classes
        self.rho = rho
        self.per_class = [
            {
                'protos': torch.empty(0, 64),
                'seen': 0,
                'radius': 0.0
            } for _ in range(num_classes)
        ]

    def should_keep(self, z_norm: torch.Tensor, y: int) -> bool:
        st = self.per_class[y]
        st['seen'] += 1
        quota = int(max(1, round(self.rho * math.log(max(2, st['seen'])))))
        P = st['protos'].size(0)
        if P < quota:
            self._add_proto(y, z_norm)
            return True
        d2 = torch.cdist(z_norm.view(1, -1), st['protos']).squeeze(0)
        min_d = float(d2.min().item()) if d2.numel() > 0 else float('inf')
        if min_d > st['radius']:
            idx = int(d2.argmin().item()) if d2.numel() > 0 else 0
            with torch.no_grad():
                st['protos'][idx] = z_norm.detach().cpu()
            st['radius'] = running_mean(st['radius'], min_d, P + 1)
            return True
        return False

    def _add_proto(self, y: int, z_norm: torch.Tensor):
        st = self.per_class[y]
        zn = z_norm.detach().cpu().view(1, -1)
        if st['protos'].numel() == 0:
            st['protos'] = zn
            st['radius'] = 0.0
        else:
            st['protos'] = torch.cat([st['protos'], zn], dim=0)
            if st['protos'].size(0) > 1:
                d = torch.cdist(st['protos'], st['protos'])
                d[d == 0] = d.max()
                st['radius'] = float(d.min(dim=1).values.mean().item())


# ------------------------------
# Episodic Memory Manager
# ------------------------------

@dataclass
class MemoryItem:
    codes: torch.ByteTensor  # [M]
    label: int               # uint8 range recommended
    gen_id: int              # generation id used to encode


class EpisodicMemory:
    def __init__(self, cap_bytes: int, M: int, include_gen_id: bool = True):
        self.cap_bytes = cap_bytes
        self.items: List[MemoryItem] = []
        self.M = M
        self.include_gen_id = include_gen_id
        self.bytes_used = 0

    def _bytes_per_item(self) -> int:
        return self.M + 1 + (1 if self.include_gen_id else 0)

    def can_add(self) -> bool:
        return (self.bytes_used + self._bytes_per_item()) <= self.cap_bytes

    def add(self, codes: torch.ByteTensor, label: int, gen_id: int):
        if not self.can_add():
            return False
        self.items.append(MemoryItem(codes.cpu().clone().view(-1), int(label), int(gen_id)))
        self.bytes_used += self._bytes_per_item()
        return True

    def sample(self, K: int, device: torch.device) -> Tuple[torch.ByteTensor, torch.LongTensor, torch.LongTensor]:
        if len(self.items) == 0 or K <= 0:
            return torch.empty(0, self.M, dtype=torch.uint8, device=device), \
                   torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)
        idx = np.random.choice(len(self.items), size=min(K, len(self.items)), replace=False)
        codes = torch.stack([self.items[i].codes for i in idx])
        labels = torch.tensor([self.items[i].label for i in idx], dtype=torch.long)
        gens = torch.tensor([self.items[i].gen_id for i in idx], dtype=torch.long)
        return codes.to(device), labels.to(device), gens.to(device)


# ------------------------------
# Budget Controller
# ------------------------------

class BudgetController:
    def __init__(self, C_per_task: int = 400):
        self.C = C_per_task
        self.remaining = C_per_task

    def begin_task(self, C_per_task: Optional[int] = None):
        self.C = self.C if C_per_task is None else C_per_task
        self.remaining = self.C

    def decide_K(self, requested_K: int) -> int:
        if self.remaining <= 0:
            return 0
        if self.remaining < 10:
            return max(0, min(requested_K, 8))
        return requested_K

    def register_backprop(self, units: int = 1):
        self.remaining = max(0, self.remaining - units)


# ------------------------------
# Training/Evaluation Utilities
# ------------------------------

@dataclass
class TrainConfig:
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    epochs_per_task: int = 1
    warmup_epochs_first_task: int = 1
    freeze_after_warmup: bool = True
    tau_recon: float = 0.1
    rho_gis: float = 1.0
    replay_K: int = 64
    budget_C: int = 400
    mem_cap_mb: float = 5.0


@dataclass
class RunStats:
    acc_per_task: List[float] = field(default_factory=list)
    forget_per_task: List[float] = field(default_factory=list)
    memory_mb_per_task: List[float] = field(default_factory=list)
    loss_curve: List[float] = field(default_factory=list)
    gen_count_per_task: List[int] = field(default_factory=list)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_plot(fig_path: str):
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def train_propqem(task_stream, num_classes: int, cfg: TrainConfig, device: torch.device, seed: int = 0,
                  save_dir: str = '.research/iteration1/images', verbose: bool = True) -> RunStats:
    set_seed(seed)
    _ensure_dir(save_dir)

    backbone = TinyConvBackbone(out_dim=256).to(device)
    residual = ResidualMLP(in_dim=256, hidden=128, out_dim=64).to(device)
    decoder = DeltaDecoder(in_dim=64, hidden=128, out_dim=256).to(device)
    classifier = Classifier(in_dim=256, num_classes=num_classes).to(device)

    params = list(residual.parameters()) + list(decoder.parameters()) + list(classifier.parameters())
    params += [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    pq = ProductQuantiser(D=64, M=8, Ks=256, device=device).to(device)
    gis = GISState(num_classes=num_classes, rho=cfg.rho_gis)

    mem_cap_bytes = int(cfg.mem_cap_mb * 1024 * 1024)
    memory = EpisodicMemory(cap_bytes=mem_cap_bytes, M=pq.M, include_gen_id=True)

    budget = BudgetController(C_per_task=cfg.budget_C)

    stats = RunStats()
    acc_matrix: List[List[float]] = []

    global_step = 0
    for task_id, train_loader, val_loader, test_loader, cls in task_stream:
        if verbose:
            print(f"\n===== Task {task_id} | Classes: {cls} =====")
        budget.begin_task(cfg.budget_C)
        is_first_task = (task_id == 0)

        total_epochs = cfg.epochs_per_task + (cfg.warmup_epochs_first_task if is_first_task else 0)
        for epoch in range(total_epochs):
            backbone.train(); residual.train(); decoder.train(); classifier.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = to_device(xb, device)
                yb = to_device(yb, device)

                f = backbone(xb)
                z = residual(f)

                with torch.no_grad():
                    codes = pq.encode(z.detach())
                kept = 0
                for i in range(z.size(0)):
                    z_norm = F.normalize(z[i], dim=0)
                    label_i = int(yb[i].item())
                    if gis.should_keep(z_norm, label_i):
                        if memory.can_add():
                            ok = memory.add(codes[i].detach().cpu(), label_i, pq.current_generation_id())
                            if ok:
                                kept += 1

                # Replay under budget
                K = budget.decide_K(cfg.replay_K)
                mem_codes, mem_labels, mem_gens = memory.sample(K, device)
                if mem_codes.size(0) > 0:
                    z_replay = pq.decode_mixed_gens(mem_codes, mem_gens)
                    f_recon = decoder(z_replay)
                    logits_rep = classifier(f_recon)
                    loss_replay = F.cross_entropy(logits_rep, mem_labels)
                else:
                    loss_replay = torch.tensor(0.0, device=device)

                logits_cur = classifier(f)
                loss_cls = F.cross_entropy(logits_cur, yb)

                with torch.no_grad():
                    codes_tmp = pq.encode(z.detach())
                    z_pq = pq.decode(codes_tmp, pq.current_generation_id())
                loss_geom = cosine_error(z, z_pq).mean()

                loss = loss_cls + 0.5 * loss_replay + 0.1 * loss_geom
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                epoch_loss += float(loss.item())
                budget.register_backprop(1)
                global_step += 1

            epoch_loss /= max(1, len(train_loader))
            stats.loss_curve.append(epoch_loss)
            if verbose:
                print(f"Task {task_id} Epoch {epoch} | loss={epoch_loss:.4f} | kept={kept} | mem={bytes_to_mb(memory.bytes_used):.3f} MB | gen={pq.generation_count()}")

            if is_first_task and cfg.freeze_after_warmup and epoch + 1 == cfg.warmup_epochs_first_task:
                if verbose:
                    print("Freezing bottom-3 blocks of backbone after warm-up.")
                backbone.freeze_low_blocks()

            # Curriculum codebook growth check
            with torch.no_grad():
                z_list = []
                for xv, _ in val_loader:
                    xv = to_device(xv, device)
                    f_val = backbone(xv)
                    z_val = residual(f_val)
                    z_list.append(z_val)
                    if len(z_list) * val_loader.batch_size >= 256:
                        break
                if len(z_list) > 0:
                    z_sketch = torch.cat(z_list, dim=0)
                    recon_err = pq.recon_error(z_sketch).item()
                    if verbose:
                        print(f"Validation PQ recon error (cosine): {recon_err:.4f}")
                    if recon_err > cfg.tau_recon:
                        if verbose:
                            print("Recon error exceeds tau; adding new PQ generation.")
                        pq.add_new_generation(init_means=z_sketch.detach())

        # Evaluate on all seen tasks so far
        acc_this_task = []
        for _, _, _, test_loader_j, _ in task_stream[:task_id+1]:
            acc_j, _, _ = evaluate(backbone, classifier, test_loader_j, device, num_classes)
            acc_this_task.append(acc_j)
        acc_matrix.append(acc_this_task)

        acc_current_task = acc_this_task[-1] if len(acc_this_task) else 0.0
        stats.acc_per_task.append(acc_current_task)
        fgt_list = compute_forgetting(acc_matrix)
        avg_fgt = float(np.mean(fgt_list)) if len(fgt_list) else 0.0
        stats.forget_per_task.append(avg_fgt)
        total_bytes = memory.bytes_used + pq.bytes_for_codebooks()
        stats.memory_mb_per_task.append(bytes_to_mb(total_bytes))
        stats.gen_count_per_task.append(pq.generation_count())
        if verbose:
            print(f"After Task {task_id}: ACC={acc_current_task*100:.2f}% | AvgFGT={avg_fgt*100:.2f}% | Memory={bytes_to_mb(total_bytes):.3f} MB | Generations={pq.generation_count()}")

    # Final confusion matrix on last task
    last_test_loader = task_stream[-1][3]
    acc_last, preds, gts = evaluate(backbone, classifier, last_test_loader, device, num_classes)
    cm = compute_confusion_matrix(preds, gts, num_classes)

    # Plots
    plt.figure(figsize=(5,3))
    plt.plot(stats.loss_curve, label='train_loss')
    plt.xlabel('Epochs (cumulative)')
    plt.ylabel('Loss')
    plt.title('Training Loss (ProPqEM)')
    plt.legend()
    _save_plot(os.path.join(save_dir, 'training_loss_propqem.pdf'))

    plt.figure(figsize=(5,3))
    plt.plot(list(range(len(stats.acc_per_task))), stats.acc_per_task, marker='o')
    plt.xlabel('Task')
    plt.ylabel('ACC (current task)')
    plt.title('Accuracy per Task (ProPqEM)')
    _save_plot(os.path.join(save_dir, 'accuracy_propqem.pdf'))

    plt.figure(figsize=(5,3))
    plt.plot(stats.memory_mb_per_task, marker='s')
    plt.xlabel('Task')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Footprint (ProPqEM)')
    _save_plot(os.path.join(save_dir, 'memory_footprint_propqem.pdf'))

    plt.figure(figsize=(6,5))
    cm_disp = np.log1p(cm)
    sns.heatmap(cm_disp, cmap='viridis')
    plt.title('Confusion Matrix (log1p) – ProPqEM')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    _save_plot(os.path.join(save_dir, 'confusion_matrix_propqem.pdf'))

    return stats


def train_baseline_raw_feature(task_stream, num_classes: int, cfg: TrainConfig, device: torch.device, seed: int = 0,
                               save_dir: str = '.research/iteration1/images', verbose: bool = True) -> RunStats:
    set_seed(seed)
    _ensure_dir(save_dir)

    backbone = TinyConvBackbone(out_dim=256).to(device)
    classifier = Classifier(in_dim=256, num_classes=num_classes).to(device)
    opt = torch.optim.SGD(list(backbone.parameters()) + list(classifier.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    mem_cap_bytes = int(cfg.mem_cap_mb * 1024 * 1024)
    stored_features: List[Tuple[torch.Tensor, int]] = []
    bytes_used = 0

    stats = RunStats()
    acc_matrix: List[List[float]] = []

    for task_id, train_loader, _, test_loader, cls in task_stream:
        if verbose:
            print(f"\n[Baseline] Task {task_id} | Classes: {cls}")
        for epoch in range(cfg.epochs_per_task):
            backbone.train(); classifier.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = to_device(xb, device)
                yb = to_device(yb, device)
                f = backbone(xb)
                logits = classifier(f)
                loss = F.cross_entropy(logits, yb)

                if len(stored_features) > 0:
                    idx = np.random.choice(len(stored_features), size=min(cfg.replay_K, len(stored_features)), replace=False)
                    f_rep = torch.stack([stored_features[i][0] for i in idx]).to(device)
                    y_rep = torch.tensor([stored_features[i][1] for i in idx], dtype=torch.long, device=device)
                    logits_rep = classifier(f_rep)
                    loss = loss + 0.5 * F.cross_entropy(logits_rep, y_rep)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                epoch_loss += float(loss.item())

                # Store some features under memory cap (float32)
                f_det = f.detach().cpu()
                for i in range(f_det.size(0)):
                    item_bytes = count_bytes_tensor(f_det[i]) + 1
                    if bytes_used + item_bytes <= mem_cap_bytes:
                        stored_features.append((f_det[i].clone(), int(yb[i].item())))
                        bytes_used += item_bytes

            epoch_loss /= max(1, len(train_loader))
            stats.loss_curve.append(epoch_loss)
            if verbose:
                print(f"[Baseline] Task {task_id} Epoch {epoch} | loss={epoch_loss:.4f} | mem={bytes_to_mb(bytes_used):.3f} MB")

        # Eval on seen tasks
        acc_this_task = []
        for _, _, _, test_loader_j, _ in task_stream[:task_id+1]:
            acc_j, _, _ = evaluate(backbone, classifier, test_loader_j, device, num_classes)
            acc_this_task.append(acc_j)
        acc_matrix.append(acc_this_task)

        acc_current_task = acc_this_task[-1] if len(acc_this_task) else 0.0
        stats.acc_per_task.append(acc_current_task)
        fgt_list = compute_forgetting(acc_matrix)
        avg_fgt = float(np.mean(fgt_list)) if len(fgt_list) else 0.0
        stats.forget_per_task.append(avg_fgt)
        stats.memory_mb_per_task.append(bytes_to_mb(bytes_used))
        if verbose:
            print(f"[Baseline] After Task {task_id}: ACC={acc_current_task*100:.2f}% | AvgFGT={avg_fgt*100:.2f}% | Memory={bytes_to_mb(bytes_used):.3f} MB")

    # Plots
    plt.figure(figsize=(5,3))
    plt.plot(stats.loss_curve, label='train_loss')
    plt.xlabel('Epochs (cumulative)')
    plt.ylabel('Loss')
    plt.title('Training Loss (Baseline Raw-Feature ER)')
    plt.legend()
    _save_plot(os.path.join(save_dir, 'training_loss_baseline.pdf'))

    plt.figure(figsize=(5,3))
    plt.plot(list(range(len(stats.acc_per_task))), stats.acc_per_task, marker='o')
    plt.xlabel('Task')
    plt.ylabel('ACC (current task)')
    plt.title('Accuracy per Task (Baseline)')
    _save_plot(os.path.join(save_dir, 'accuracy_baseline.pdf'))

    plt.figure(figsize=(5,3))
    plt.plot(stats.memory_mb_per_task, marker='s')
    plt.xlabel('Task')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Footprint (Baseline)')
    _save_plot(os.path.join(save_dir, 'memory_footprint_baseline.pdf'))

    return stats
