# -*- coding: utf-8 -*-
import os
import time
import json
import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision

from .preprocess import set_seed

# Ensure non-interactive backend for servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------
# Model definitions
# ------------------------------
class ResNet18Feature(nn.Module):
    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.base = base

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # [B,512]

    def freeze_early(self, L: int = 2):
        groups = [self.base.conv1, self.base.bn1, self.base.layer1, self.base.layer2]
        for g in groups[:max(1, min(L, len(groups)) )]:
            for p in g.parameters():
                p.requires_grad = False


class Reducer(nn.Module):
    def __init__(self, d_in=512, d_red=192):
        super().__init__()
        self.fc = nn.Linear(d_in, d_red)
        self.ln = nn.LayerNorm(d_red)
    def forward(self, x):
        return self.ln(self.fc(x))


class AdaptorH(nn.Module):
    def __init__(self, d_red):
        super().__init__()
        self.lin = nn.Linear(d_red, d_red)
    def forward(self, z):
        return self.lin(z)


class TailMLP(nn.Module):
    def __init__(self, d_in, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(inplace=True), nn.Linear(256, num_classes)
        )
    def forward(self, z):
        return self.mlp(z)


class CLNet(nn.Module):
    def __init__(self, num_classes, d_red):
        super().__init__()
        self.f0 = ResNet18Feature()
        self.reducer = Reducer(512, d_red)
        self.adaptor = AdaptorH(d_red)
        self.f1 = TailMLP(d_red, num_classes)

    def forward(self, x):
        feat = self.f0(x)
        z = self.reducer(feat)
        zh = self.adaptor(z)
        logits = self.f1(zh)
        return logits


# ------------------------------
# FREQUENT components
# ------------------------------
class OnlinePQ(nn.Module):
    def __init__(self, d_red=192, M=8, K=128, ema_decay=0.99, eps=1e-5):
        super().__init__()
        assert d_red % M == 0, "d_red must be divisible by M"
        self.d_red = d_red
        self.M = M
        self.K = K
        self.d_sub = d_red // M
        cb = torch.randn(M, K, self.d_sub) * 0.1
        self.codebook = nn.Parameter(cb)
        self.register_buffer('ema_count', torch.zeros(M, K))
        self.register_buffer('ema_sum', torch.zeros(M, K, self.d_sub))
        self.ema_decay = ema_decay
        self.eps = eps

    @torch.no_grad()
    def _assign(self, z):
        B = z.shape[0]
        z_ = z.view(B, self.M, self.d_sub)
        z2 = (z_ ** 2).sum(-1, keepdim=True)
        c2 = (self.codebook ** 2).sum(-1).unsqueeze(0)
        zc = torch.einsum('bmd, mkd -> bmk', z_, self.codebook)
        dist = z2 - 2 * zc + c2
        codes = dist.argmin(-1)
        return codes

    @torch.no_grad()
    def update_ema(self, z, codes):
        B = z.size(0)
        z_ = z.view(B, self.M, self.d_sub)
        ema_count = torch.zeros_like(self.ema_count)
        ema_sum = torch.zeros_like(self.ema_sum)
        for m in range(self.M):
            idx = codes[:, m]
            oh = F.one_hot(idx, num_classes=self.K).to(z.dtype)
            ema_count[m] += oh.sum(0)
            sums = torch.einsum('bk, bd -> kd', oh, z_[:, m, :])
            ema_sum[m] += sums
        self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * ema_count
        self.ema_sum = self.ema_decay * self.ema_sum + (1 - self.ema_decay) * ema_sum
        count = (self.ema_count + self.eps)
        new_cb = self.ema_sum / count.unsqueeze(-1)
        new_cb = torch.where(torch.isnan(new_cb), self.codebook, new_cb)
        self.codebook.data.copy_(new_cb)

    def quantize(self, z):
        with torch.no_grad():
            codes = self._assign(z)
        return codes

    def dequantize(self, codes):
        B = codes.size(0)
        out = []
        for m in range(self.M):
            c = self.codebook[m]
            idx = codes[:, m]
            out.append(c[idx])
        zq = torch.cat(out, dim=-1)
        return zq


@dataclass
class BudgetConfig:
    name: str
    d_red: int
    M: int
    K: int
    meta_bytes_per_entry: int = 4
    def codebook_bytes(self):
        d_sub = self.d_red // self.M
        return self.M * self.K * d_sub * 4
    def bytes_per_entry(self):
        index_bytes = self.M * (1 if self.K <= 256 else 2)
        return index_bytes + self.meta_bytes_per_entry


BUDGETS = {
    '50kB': BudgetConfig('50kB', d_red=96,  M=6, K=64),
    '200kB': BudgetConfig('200kB', d_red=192, M=8, K=128),
    '1MB': BudgetConfig('1MB', d_red=256, M=8, K=256),
}


class FrequentMemory:
    def __init__(self, budget: BudgetConfig, total_budget_bytes: int, adaptor: AdaptorH, device: str):
        self.budget = budget
        self.total_budget_bytes = total_budget_bytes
        self.adaptor = adaptor
        self.codebook = OnlinePQ(d_red=budget.d_red, M=budget.M, K=budget.K).to(device)
        self.bytes_per_entry = budget.bytes_per_entry()
        overhead = self.codebook_bytes() + self.adaptor_bytes()
        remain = max(0, total_budget_bytes - overhead)
        self.capacity = max(1, remain // self.bytes_per_entry)
        self.codes = torch.zeros((self.capacity, budget.M), dtype=torch.uint8, device=device)
        self.labels = torch.zeros((self.capacity,), dtype=torch.int16, device=device)
        self.age = torch.zeros((self.capacity,), dtype=torch.int16, device=device)
        self.relo_nonpos = torch.zeros((self.capacity,), dtype=torch.int16, device=device)
        self.valid = torch.zeros((self.capacity,), dtype=torch.bool, device=device)
        self.ptr = 0
        self.size = 0
        self.relo_window = 200

    def codebook_bytes(self):
        return self.budget.codebook_bytes()
    def adaptor_bytes(self):
        n = sum(p.numel() for p in self.adaptor.parameters())
        return n * 4
    def total_bytes(self):
        return self.codebook_bytes() + self.adaptor_bytes() + self.size * self.bytes_per_entry
    def memory_breakdown(self):
        return {
            'codebook_B': int(self.codebook_bytes()),
            'adaptor_B': int(self.adaptor_bytes()),
            'entries_B': int(self.size * self.bytes_per_entry),
            'bytes_per_entry': int(self.bytes_per_entry),
            'capacity_entries': int(self.capacity),
        }

    @torch.no_grad()
    def insert_batch(self, z, y):
        codes = self.codebook.quantize(z)
        self.codebook.update_ema(z, codes)
        B = codes.size(0)
        for i in range(B):
            pos = self.ptr
            self.codes[pos] = codes[i].to(torch.uint8)
            self.labels[pos] = int(y[i].item())
            self.age[pos] = 0
            self.relo_nonpos[pos] = 0
            self.valid[pos] = True
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    @torch.no_grad()
    def sample(self, bs):
        if self.size == 0:
            return None, None, None
        bs = min(bs, self.size)
        idxs = torch.randint(0, self.size, (bs,), device=self.codes.device)
        codes = self.codes[idxs].long()
        labels = self.labels[idxs].long()
        return idxs, codes, labels

    @torch.no_grad()
    def update_relo(self, idxs, loss_online, loss_target):
        relo = loss_online - loss_target
        nonpos = (relo <= 0).to(self.relo_nonpos.dtype)
        self.relo_nonpos[idxs] += nonpos
        self.age[idxs] += 1
        evict_mask = (self.relo_nonpos > self.relo_window) & self.valid
        if evict_mask.any():
            evict_idxs = torch.where(evict_mask)[0]
            self.valid[evict_idxs] = False
            self.relo_nonpos[evict_idxs] = 0
            self.age[evict_idxs] = 0


class ERRingBuffer:
    def __init__(self, total_budget_bytes: int):
        self.bytes_per_image = 32*32*3
        self.bytes_per_entry = self.bytes_per_image + 2
        self.capacity = max(1, total_budget_bytes // self.bytes_per_entry)
        self.imgs = None
        self.labels = None
        self.ptr = 0
        self.size = 0

    @torch.no_grad()
    def insert_batch(self, x, y):
        B = x.size(0)
        device = x.device
        if self.imgs is None:
            self.imgs = torch.zeros((self.capacity, *x.shape[1:]), dtype=torch.uint8, device=device)
            self.labels = torch.zeros((self.capacity,), dtype=torch.int16, device=device)
        x_uint8 = (x.clamp(0,1) * 255.0).to(torch.uint8)
        for i in range(B):
            pos = self.ptr
            self.imgs[pos] = x_uint8[i]
            self.labels[pos] = int(y[i].item())
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    @torch.no_grad()
    def sample(self, bs):
        if self.size == 0:
            return None, None
        bs = min(bs, self.size)
        idxs = torch.randint(0, self.size, (bs,), device=self.labels.device)
        x = self.imgs[idxs].float() / 255.0
        y = self.labels[idxs].long()
        return x, y


# ------------------------------
# Compute calibration
# ------------------------------

def _measure_step_time(step_fn, warmup=2, steps=4):
    times = []
    for _ in range(warmup):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(steps):
        t0 = time.perf_counter()
        step_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times))


def calibrate_replay_ratio(model: CLNet, num_classes: int, method: str, alpha: float, bs: int, device: str) -> Tuple[float, float, float, float]:
    loss_fn = nn.CrossEntropyLoss()
    x = torch.rand(bs, 3, 32, 32, device=device)
    y = torch.randint(0, num_classes, (bs,), device=device)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)

    def step_online():
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

    t_online = _measure_step_time(step_online)

    xr = torch.rand(bs, 3, 32, 32, device=device)
    yr = torch.randint(0, num_classes, (bs,), device=device)

    def step_replay_full():
        opt.zero_grad(set_to_none=True)
        logits = model(xr)
        loss = loss_fn(logits, yr)
        loss.backward()
        opt.step()

    t_rep_full = _measure_step_time(step_replay_full)

    if method == 'FREQUENT':
        zr = torch.rand(bs, model.reducer.fc.out_features, device=device)
        def step_replay_bypass():
            opt.zero_grad(set_to_none=True)
            logits = model.f1(model.adaptor(zr))
            loss = loss_fn(logits, yr)
            loss.backward()
            opt.step()
        t_rep_method = _measure_step_time(step_replay_bypass)
    elif method == 'SEER':
        zr = torch.rand(bs, model.reducer.fc.out_features, device=device)
        def step_replay_bypass():
            opt.zero_grad(set_to_none=True)
            logits = model.f1(model.adaptor(zr))
            loss = loss_fn(logits, yr)
            loss.backward()
            opt.step()
        t_rep_method = _measure_step_time(step_replay_bypass)
    else:
        t_rep_method = t_rep_full

    rhs = alpha * (t_online + 0.25 * t_rep_full)
    num = rhs - t_online
    den = max(1e-9, t_rep_method)
    r = float(max(num / den, 0.05))
    return r, t_online, t_rep_method, t_rep_full


# ------------------------------
# Training loop
# ------------------------------

def train_continual(
    exp_name: str,
    out_image_dir: str,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    num_classes_total: int,
    methods: Tuple[str,...],
    budget_tag: str,
    alpha: float,
    seed: int,
    epochs_per_task: int,
    classes_per_task: int,
    batch_size: int,
    quick_test: bool,
) -> Dict:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sns.set_style('whitegrid')
    os.makedirs(out_image_dir, exist_ok=True)

    results = {
        'per_method': {}
    }

    budget = BUDGETS[budget_tag]

    for method in methods:
        set_seed(seed)
        model = CLNet(num_classes=num_classes_total, d_red=budget.d_red).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs_per_task*len(train_loaders)))
        loss_fn = nn.CrossEntropyLoss()

        memory_total_bytes = 50_000 if budget_tag=='50kB' else (200_000 if budget_tag=='200kB' else 1_000_000)
        if method == 'FREQUENT':
            mem = FrequentMemory(budget, memory_total_bytes, adaptor=model.adaptor, device=device)
        elif method == 'ER':
            mem = ERRingBuffer(memory_total_bytes)
        else:
            raise ValueError('Only FREQUENT and ER are supported in this minimal experiment runner')

        r, t_online, t_rep_method, t_rep_full = calibrate_replay_ratio(model, num_classes_total, method, alpha, bs=batch_size, device=device)

        # Target model EMA for ReLo (FREQUENT only)
        target_model = copy.deepcopy(model).to(device)
        for p in target_model.parameters():
            p.requires_grad = False
        ema_tau = 0.99

        acc_matrix = np.zeros((len(train_loaders), len(train_loaders)))
        losses_curve = []

        for task_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
            model.train()
            for epoch in range(epochs_per_task):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)

                    # online update
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    losses_curve.append(float(loss.item()))

                    # insert to memory
                    with torch.no_grad():
                        feat512 = model.f0(x)
                        z = model.reducer(feat512)
                        if method == 'FREQUENT':
                            mem.insert_batch(z, y)
                        else:
                            mem.insert_batch(x, y)

                    # replay
                    if random.random() < r:
                        if method == 'FREQUENT':
                            idxs, codes, yr = mem.sample(bs=min(batch_size, 64))
                            if codes is not None:
                                zr = mem.codebook.dequantize(codes)
                                optimizer.zero_grad(set_to_none=True)
                                logits_r = model.f1(model.adaptor(zr))
                                loss_r = loss_fn(logits_r, yr)
                                (0.5 * loss_r).backward()
                                optimizer.step()
                                with torch.no_grad():
                                    logits_online = model.f1(model.adaptor(zr))
                                    logits_target = target_model.f1(target_model.adaptor(zr))
                                    L_on = F.cross_entropy(logits_online, yr, reduction='none')
                                    L_tg = F.cross_entropy(logits_target, yr, reduction='none')
                                mem.update_relo(idxs, L_on, L_tg)
                        else:
                            xr, yr = mem.sample(bs=min(batch_size, 64))
                            if xr is not None:
                                optimizer.zero_grad(set_to_none=True)
                                logits_r = model(xr)
                                loss_r = loss_fn(logits_r, yr)
                                (0.5 * loss_r).backward()
                                optimizer.step()

                    # EMA update for FREQUENT
                    with torch.no_grad():
                        for p_t, p in zip(target_model.parameters(), model.parameters()):
                            p_t.copy_(ema_tau * p_t + (1-ema_tau) * p)

            # Freeze early layers after each task for FREQUENT
            if method == 'FREQUENT':
                model.f0.freeze_early(L=2)

            # Eval on seen tasks
            model.eval()
            with torch.no_grad():
                for j in range(task_id+1):
                    correct = 0
                    total = 0
                    for xx, yy in test_loaders[j]:
                        xx, yy = xx.to(device), yy.to(device)
                        logit = model(xx)
                        pred = logit.argmax(1)
                        correct += (pred == yy).sum().item()
                        total += yy.numel()
                    acc_matrix[task_id, j] = correct / max(1, total)

        # Metrics
        final_avg_acc = float(np.mean(acc_matrix[len(train_loaders)-1, :len(train_loaders)]))
        # forgetting
        T = len(train_loaders)
        Fvals = []
        for j in range(T):
            prev = acc_matrix[:j, j] if j > 0 else np.array([0.0])
            if len(prev) == 0:
                Fvals.append(0.0)
            else:
                Fvals.append(max(prev) - acc_matrix[T-1, j])
        forgetting = float(np.mean(Fvals))

        # Plots -> PDF
        plt.figure(figsize=(4,3))
        plt.plot(losses_curve, label='train loss')
        plt.xlabel('step'); plt.ylabel('loss'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_image_dir, f"{exp_name}_loss_{method}_{budget_tag}_seed{seed}_a{alpha}.pdf"), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(4,3))
        plt.plot(np.mean(acc_matrix[:T, :T], axis=1), label='avg acc over seen tasks')
        plt.xlabel('after task'); plt.ylabel('accuracy'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_image_dir, f"{exp_name}_acc_{method}_{budget_tag}_seed{seed}_a{alpha}.pdf"), bbox_inches='tight')
        plt.close()

        results['per_method'][method] = {
            'final_avg_acc': final_avg_acc,
            'forgetting': forgetting,
            'acc_matrix': acc_matrix.tolist(),
            'replay_ratio': r,
            't_online_s': t_online,
            't_rep_method_s': t_rep_method,
            't_rep_full_s': t_rep_full,
            'memory_breakdown': mem.memory_breakdown() if hasattr(mem, 'memory_breakdown') else {
                'bytes_per_entry': getattr(mem, 'bytes_per_entry', -1),
                'capacity_entries': getattr(mem, 'capacity', -1)
            }
        }

        # save a small checkpoint (for inspection)
        os.makedirs('models', exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'config': {
            'method': method,
            'budget_tag': budget_tag,
            'seed': seed,
            'alpha': alpha
        }}, os.path.join('models', f"{exp_name}_{method}_{budget_tag}_seed{seed}_a{alpha}.pt"))

    return results
