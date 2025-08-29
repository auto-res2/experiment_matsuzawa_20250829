#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training components for ProPqEM and baselines.
Includes:
- TinyBackbone -> f (256D)
- ResidualEncoder -> z (64D)
- DeltaDecoder: z -> f
- ClassifierHead
- ProductQuantizer with multi-generation support
- EpisodicMemory with GIS-like selection
- ProPqEMLearner and ERFeatureLearner training loops (evaluation done in evaluate.py)

Notes:
- All modules use lightweight operations suitable for NVIDIA T4 (16GB) and quick smoke tests.
- Memory accounting and PDF figure saving handled in src.main.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Optional FLOPs counter
try:
    from fvcore.nn import FlopCountAnalysis  # type: ignore
    _FLOPS_OK = True
except Exception:
    _FLOPS_OK = False


# -----------------------------
# Utilities
# -----------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_n * b_n).sum(dim=-1)


# -----------------------------
# Models
# -----------------------------

class TinyBackbone(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )
        self.frozen = False

    def freeze_low(self):
        for m in [self.conv1, self.conv2, self.conv3]:
            for p in m.parameters():
                p.requires_grad = False
        self.frozen = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        f = self.head(x)
        return f


class ResidualEncoder(nn.Module):
    def __init__(self, in_dim=256, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim)
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class DeltaDecoder(nn.Module):
    def __init__(self, in_dim=64, hidden=256, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=256, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.fc(f)


# -----------------------------
# Product Quantizer (with generations)
# -----------------------------

class ProductQuantizer:
    def __init__(self, M=8, d_sub=8, K=256, alpha=0.05, tau=0.07, device: str = "cpu"):
        self.M = M
        self.d_sub = d_sub
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.device = device
        self.generations: List[torch.Tensor] = []  # each: [M, K, d_sub]
        self.gen_usage: List[List[torch.Tensor]] = []  # per-gen usage counts per subspace [M, K]
        self.latest_moving_mse = 0.0
        self.mse_ma_alpha = 0.1

        # Optional KMeans
        try:
            from sklearn.cluster import MiniBatchKMeans  # type: ignore
            self._SKLEARN_OK = True
            self._MiniBatchKMeans = MiniBatchKMeans
        except Exception:
            self._SKLEARN_OK = False
            self._MiniBatchKMeans = None

    @property
    def z_dim(self) -> int:
        return self.M * self.d_sub

    def _init_kmeans_sub(self, sketch: np.ndarray) -> torch.Tensor:
        N = sketch.shape[0]
        codebooks = []
        for m in range(self.M):
            X = sketch[:, m * self.d_sub:(m + 1) * self.d_sub]
            if self._SKLEARN_OK and N >= self.K:
                km = self._MiniBatchKMeans(n_clusters=self.K, batch_size=min(1024, N), n_init=1, max_iter=50, verbose=0)
                km.fit(X)
                C = km.cluster_centers_.astype(np.float32)
            else:
                idx = np.random.choice(N, size=min(self.K, N), replace=False)
                C = np.zeros((self.K, self.d_sub), dtype=np.float32)
                C[:len(idx)] = X[idx].astype(np.float32)
                if len(idx) < self.K:
                    C[len(idx):] = X[np.random.choice(N, size=self.K - len(idx), replace=True)].astype(np.float32)
            codebooks.append(torch.from_numpy(C))
        cb = torch.stack(codebooks, dim=0).to(self.device)  # [M, K, d_sub]
        return cb

    def warm_start(self, sketch_z: torch.Tensor):
        sketch_np = sketch_z.detach().cpu().numpy()
        cb = self._init_kmeans_sub(sketch_np)
        self.generations = [cb]
        self.gen_usage = [[torch.zeros(self.K, dtype=torch.long, device=self.device) for _ in range(self.M)]]
        print(f"[PQ] Warm-started generation-0 codebooks with shape {tuple(cb.shape)}")

    def add_generation(self, sketch_z: torch.Tensor):
        sketch_np = sketch_z.detach().cpu().numpy()
        cb = self._init_kmeans_sub(sketch_np)
        self.generations.append(cb)
        self.gen_usage.append([torch.zeros(self.K, dtype=torch.long, device=self.device) for _ in range(self.M)])
        print(f"[PQ] Added new generation {len(self.generations)-1} with codebook shape {tuple(cb.shape)}")

    def _assign_codes_latest(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(self.generations) > 0, "PQ not initialised. Call warm_start() first."
        cb = self.generations[-1]  # [M, K, d_sub]
        B = z.shape[0]
        z_sub = z.view(B, self.M, self.d_sub)
        codes = torch.empty(B, self.M, dtype=torch.long, device=z.device)
        rec_sub = torch.empty_like(z_sub)
        for m in range(self.M):
            C = cb[m]  # [K, d_sub]
            zs = z_sub[:, m].unsqueeze(1)  # [B,1,d]
            dist2 = ((zs - C.unsqueeze(0)) ** 2).sum(dim=-1)  # [B, K]
            idx = torch.argmin(dist2, dim=1)
            codes[:, m] = idx
            rec_sub[:, m] = C[idx]
        rec_z = rec_sub.reshape(B, -1)
        return codes, rec_z

    def encode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codes, rec_z = self._assign_codes_latest(z)
        gens = torch.full((z.size(0),), len(self.generations) - 1, dtype=torch.long, device=z.device)
        return codes, rec_z, gens

    def decode(self, codes: torch.Tensor, gens: torch.Tensor) -> torch.Tensor:
        B, M = codes.shape
        assert M == self.M
        z_hat = torch.empty((B, self.z_dim), dtype=torch.float32, device=codes.device)
        for i in range(B):
            g = int(gens[i].item())
            cb = self.generations[g]
            z_sub = []
            for m in range(self.M):
                idx = int(codes[i, m].item())
                z_sub.append(cb[m, idx])
            z_hat[i] = torch.cat(z_sub, dim=0)
        return z_hat

    def ema_update_latest(self, z: torch.Tensor, codes: torch.Tensor):
        assert len(self.generations) > 0
        cb = self.generations[-1]
        usage_list = self.gen_usage[-1]
        B = z.shape[0]
        z_sub = z.view(B, self.M, self.d_sub)
        for m in range(self.M):
            idx = codes[:, m]
            usage_list[m].index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
            C = cb[m]
            unique_idx = idx.unique()
            for j in unique_idx:
                mask = (idx == j)
                if mask.any():
                    xj = z_sub[mask, m].mean(dim=0)
                    C[j] = (1.0 - self.alpha) * C[j] + self.alpha * xj
        with torch.no_grad():
            _, rec_z = self._assign_codes_latest(z)
            mse = F.mse_loss(rec_z, z).item()
            self.latest_moving_mse = (1 - self.mse_ma_alpha) * self.latest_moving_mse + self.mse_ma_alpha * mse

    def revive_dead_codes_latest(self, dead_thr: int = 1, z_pool: Optional[torch.Tensor] = None):
        cb = self.generations[-1]
        usage_list = self.gen_usage[-1]
        B = 0 if z_pool is None else z_pool.size(0)
        for m in range(self.M):
            usage = usage_list[m]
            dead = (usage <= dead_thr).nonzero(as_tuple=False).flatten()
            if len(dead) > 0:
                if z_pool is not None and B > 0:
                    repl = z_pool[torch.randint(0, B, (len(dead),)), m * self.d_sub:(m + 1) * self.d_sub]
                else:
                    repl = torch.randn(len(dead), self.d_sub, device=cb.device) * 0.01
                cb[m, dead] = repl
                usage[dead] = 1

    def maybe_grow(self, val_mse_threshold: float, sketch_z: torch.Tensor) -> bool:
        if self.latest_moving_mse > val_mse_threshold:
            self.add_generation(sketch_z)
            return True
        return False

    def memory_bytes_codebooks(self) -> int:
        total = 0
        for cb in self.generations:
            M, K, d = cb.shape
            total += int(M * K * d * 4)
        return total


# -----------------------------
# Episodic Memory with GIS-like selection
# -----------------------------

class EpisodicMemory:
    def __init__(self, pq: ProductQuantizer, item_bytes: int, cap_bytes: int, rho: int = 8,
                 sampler: str = "uniform", device: str = "cpu"):
        self.pq = pq
        self.item_bytes = int(item_bytes)
        self.cap_bytes = int(cap_bytes)
        self.rho = rho
        self.sampler = sampler
        self.device = device
        self.codes: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.gens: List[torch.Tensor] = []
        self.inserted = 0

    def total_items(self) -> int:
        return int(sum(c.size(0) for c in self.codes))

    def as_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.codes) == 0:
            return (torch.empty(0, self.pq.M, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device))
        return (torch.cat(self.codes, dim=0), torch.cat(self.labels, dim=0), torch.cat(self.gens, dim=0))

    def memory_bytes(self) -> int:
        codebooks_bytes = self.pq.memory_bytes_codebooks()
        num_items = self.total_items()
        return int(num_items * self.item_bytes + codebooks_bytes)

    def _per_class_indices(self, y: torch.Tensor) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for cls in y.unique().tolist():
            idx = (y == cls).nonzero(as_tuple=False).flatten()
            out[int(cls)] = idx
        return out

    def _farthest_first_select(self, z_norm: torch.Tensor, k: int) -> torch.Tensor:
        N = z_norm.size(0)
        if k >= N:
            return torch.arange(N, device=z_norm.device)
        sel = [0]
        dist = 1 - (z_norm @ z_norm[sel[0]].unsqueeze(0).T).squeeze(1)
        for _ in range(1, k):
            idx = torch.argmax(dist).item()
            sel.append(idx)
            dist = torch.minimum(dist, 1 - (z_norm @ z_norm[idx].unsqueeze(0).T).squeeze(1))
        return torch.tensor(sel, device=z_norm.device, dtype=torch.long)

    def _enforce_cap_with_gis(self):
        codebooks_bytes = self.pq.memory_bytes_codebooks()
        avail_bytes = max(0, self.cap_bytes - codebooks_bytes)
        max_items = avail_bytes // self.item_bytes if self.item_bytes > 0 else 0
        codes, labels, gens = self.as_tensors()
        N = codes.size(0)
        if N <= max_items:
            return
        with torch.no_grad():
            z_hat = self.pq.decode(codes, gens)
            z_norm = F.normalize(z_hat, dim=-1)
        cls_indices = self._per_class_indices(labels)
        selected_global = []
        per_class_target: Dict[int, int] = {}
        total_target = 0
        for cls, idx in cls_indices.items():
            Nc = idx.numel()
            k = min(Nc, max(1, int(self.rho * math.log(Nc + math.e))))
            per_class_target[cls] = k
            total_target += k
        if total_target > max_items and total_target > 0:
            scale = max_items / total_target
            for cls in per_class_target:
                per_class_target[cls] = max(1, int(per_class_target[cls] * scale))
        for cls, idx in cls_indices.items():
            k = per_class_target[cls]
            if idx.numel() <= k:
                selected_global.append(idx)
            else:
                sel_local = self._farthest_first_select(z_norm[idx], k)
                selected_global.append(idx[sel_local])
        idx_keep = torch.cat(selected_global, dim=0)
        self.codes = [codes[idx_keep]]
        self.labels = [labels[idx_keep]]
        self.gens = [gens[idx_keep]]

    def insert(self, z: torch.Tensor, y: torch.Tensor):
        codes, _, gens = self.pq.encode(z)
        self.pq.ema_update_latest(z, codes)
        self.inserted += z.size(0)
        self.codes.append(codes.detach().to(self.device))
        self.labels.append(y.detach().to(self.device).long())
        self.gens.append(gens.detach().to(self.device))
        self._enforce_cap_with_gis()

    def sample(self, K: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codes, labels, gens = self.as_tensors()
        N = codes.size(0)
        if N == 0 or K <= 0:
            return (torch.empty(0, self.pq.z_dim, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device),
                    torch.empty(0, self.pq.M, dtype=torch.long, device=self.device))
        if self.sampler == "class":
            per_class = self._per_class_indices(labels)
            take = []
            per = max(1, K // max(1, len(per_class)))
            for _, idx in per_class.items():
                if idx.numel() <= per:
                    take.append(idx)
                else:
                    perm = torch.randperm(idx.numel(), device=idx.device)[:per]
                    take.append(idx[perm])
            idx = torch.cat(take, dim=0)
            if idx.numel() > K:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[:K]]
        else:
            idx = torch.randperm(N, device=codes.device)[:min(K, N)]
        codes_s = codes[idx]
        labels_s = labels[idx]
        gens_s = gens[idx]
        z_hat = self.pq.decode(codes_s, gens_s)
        return z_hat, labels_s, codes_s


# -----------------------------
# Training configs and learners
# -----------------------------

@dataclass
class TrainConfig:
    f_dim: int = 256
    z_dim: int = 64
    M: int = 8
    d_sub: int = 8
    K: int = 256
    pq_alpha: float = 0.05
    pq_tau: float = 0.07
    mse_growth_thr: float = 0.07
    rho: int = 8
    lr: float = 0.1
    wd: float = 5e-4
    momentum: float = 0.9
    batch_size: int = 64
    epochs_warmup_task1: int = 1
    C_backprops_per_task: int = 400
    replay_K: int = 64
    replay_sampler: str = "uniform"
    lam_l2: float = 1.0
    lam_cos: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mem_cap_mb: float = 5.0
    no_gis: bool = False
    no_curriculum: bool = False
    no_f_space_loss: bool = False


class ProPqEMLearner:
    def __init__(self, num_classes: int, cfg: TrainConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.backbone = TinyBackbone(out_dim=cfg.f_dim).to(self.device)
        self.res_enc = ResidualEncoder(in_dim=cfg.f_dim, out_dim=cfg.z_dim).to(self.device)
        self.delta_dec = DeltaDecoder(in_dim=cfg.z_dim, out_dim=cfg.f_dim).to(self.device)
        self.clf = ClassifierHead(in_dim=cfg.f_dim, num_classes=num_classes).to(self.device)

        assert cfg.M * cfg.d_sub == cfg.z_dim
        self.pq = ProductQuantizer(M=cfg.M, d_sub=cfg.d_sub, K=cfg.K, alpha=cfg.pq_alpha, tau=cfg.pq_tau, device=self.device)
        item_bytes = cfg.M + 2  # M byte indices + label + gen
        self.memory = EpisodicMemory(self.pq, item_bytes=item_bytes, cap_bytes=int(cfg.mem_cap_mb * 1024 * 1024),
                                     rho=cfg.rho, sampler=cfg.replay_sampler, device=self.device)
        params = list(self.backbone.parameters()) + list(self.res_enc.parameters()) + list(self.delta_dec.parameters()) + list(self.clf.parameters())
        self.opt = torch.optim.SGD([p for p in params if p.requires_grad], lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd)
        self.backprops = 0
        self.task_id = 0
        self.history = {"task": [], "acc": [], "fg": [], "mem_bytes": [], "pq_mse": [], "gens": [], "flops_fw": [], "backprops": []}

    def _compute_flops(self, x_img: torch.Tensor, f_replay: torch.Tensor) -> float:
        if not _FLOPS_OK:
            return float('nan')
        try:
            fwd = FlopCountAnalysis(nn.Sequential(self.backbone, self.clf), x_img)
            flops_img = float(fwd.total())
        except Exception:
            flops_img = float('nan')
        try:
            fwd2 = FlopCountAnalysis(self.clf, f_replay)
            flops_replay = float(fwd2.total())
        except Exception:
            flops_replay = float('nan')
        val = (flops_img if not math.isnan(flops_img) else 0.0) + (flops_replay if not math.isnan(flops_replay) else 0.0)
        return val

    def warm_start_pq(self, loader: DataLoader, sketch_size: int = 2048):
        self.backbone.eval()
        self.res_enc.eval()
        feats = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                f = self.backbone(x)
                z = self.res_enc(f)
                feats.append(z.detach().cpu())
                if len(torch.cat(feats)) >= sketch_size:
                    break
        if len(feats) == 0:
            raise RuntimeError("Warm start loader produced no data")
        Z = torch.cat(feats)[:sketch_size].to(self.device)
        self.pq.warm_start(Z)

    def train_task(self, train_loader: DataLoader, class_ids: List[int], warmup: bool = False):
        cfg = self.cfg
        self.backprops = 0
        self.task_id += 1
        device = self.device
        self.backbone.train(); self.res_enc.train(); self.delta_dec.train(); self.clf.train()

        if warmup and self.task_id == 1:
            print("[Train] Warmup epoch for task-1 (unfrozen backbone)")
            for _ in range(cfg.epochs_warmup_task1):
                for x, y in train_loader:
                    if self.backprops >= cfg.C_backprops_per_task:
                        break
                    x, y = x.to(device), y.to(device)
                    f = self.backbone(x)
                    z = self.res_enc(f)
                    codes, z_rec, _ = self.pq.encode(z)
                    f_hat = self.delta_dec(z_rec)
                    logits_cur = self.clf(f)
                    loss_ce = F.cross_entropy(logits_cur, y)
                    loss_rec = 0.0
                    if not cfg.no_f_space_loss:
                        loss_rec = cfg.lam_l2 * F.mse_loss(f_hat, f) + cfg.lam_cos * (1 - cosine_sim(f_hat, f).mean())
                    loss = loss_ce + loss_rec
                    self.opt.zero_grad(); loss.backward(); self.opt.step()
                    self.pq.ema_update_latest(z, codes)
                    self.pq.revive_dead_codes_latest(z_pool=z)
                    self.backprops += 1
                    if self.backprops >= cfg.C_backprops_per_task:
                        break
            self.backbone.freeze_low()

        for x, y in train_loader:
            if self.backprops >= cfg.C_backprops_per_task:
                break
            x, y = x.to(device), y.to(device)
            f = self.backbone(x)
            z = self.res_enc(f)
            codes, z_rec, _ = self.pq.encode(z)
            f_hat_cur = self.delta_dec(z_rec)

            z_rep, y_rep, _ = self.memory.sample(cfg.replay_K)
            f_rep = torch.empty(0, cfg.f_dim, device=device)
            if z_rep.numel() > 0:
                f_rep = self.delta_dec(z_rep)

            _ = self._compute_flops(x, f_rep)  # not stored per step to keep runtime low

            logits_cur = self.clf(f)
            loss_ce = F.cross_entropy(logits_cur, y)
            loss_rep = 0.0
            if f_rep.numel() > 0:
                logits_rep = self.clf(f_rep)
                loss_rep = F.cross_entropy(logits_rep, y_rep)
            loss_rec = 0.0
            if not cfg.no_f_space_loss:
                loss_rec = cfg.lam_l2 * F.mse_loss(f_hat_cur, f) + cfg.lam_cos * (1 - cosine_sim(f_hat_cur, f).mean())
            loss = loss_ce + loss_rep + loss_rec

            self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.backprops += 1

            self.pq.ema_update_latest(z, codes)
            self.pq.revive_dead_codes_latest(z_pool=z)

            with torch.no_grad():
                take = min(x.size(0) // 2, 16)
                idx = torch.randperm(x.size(0), device=device)[:take]
                self.memory.insert(z[idx], y[idx])

            if not self.cfg.no_curriculum:
                self.pq.maybe_grow(cfg.mse_growth_thr, z.detach())

            if self.backprops % 50 == 0 or self.backprops == 1:
                mem_bytes = self.memory.memory_bytes()
                print(f"[Task {self.task_id}] step backprops={self.backprops} loss={loss.item():.4f} mem={mem_bytes/1e6:.3f}MB pq_mse_ma={self.pq.latest_moving_mse:.4f} gens={len(self.pq.generations)}")

        # Update history placeholders; evaluation happens in src.evaluate
        self.history["task"].append(self.task_id)
        self.history["acc"].append(0.0)
        self.history["fg"].append(0.0)
        self.history["mem_bytes"].append(self.memory.memory_bytes())
        self.history["pq_mse"].append(self.pq.latest_moving_mse)
        self.history["gens"].append(len(self.pq.generations))
        self.history["flops_fw"].append(0.0)
        self.history["backprops"].append(self.backprops)

    def get_memory_bytes(self) -> int:
        return self.memory.memory_bytes()


class ERFeatureLearner:
    def __init__(self, num_classes: int, cfg: TrainConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.backbone = TinyBackbone(out_dim=cfg.f_dim).to(self.device)
        self.clf = ClassifierHead(in_dim=cfg.f_dim, num_classes=num_classes).to(self.device)
        self.opt = torch.optim.SGD(list(self.backbone.parameters()) + list(self.clf.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd)
        self.memory_f: List[torch.Tensor] = []
        self.memory_y: List[torch.Tensor] = []
        self.f_dim = cfg.f_dim
        self.item_bytes = cfg.f_dim * 4 + 1
        self.cap_bytes = int(cfg.mem_cap_mb * 1024 * 1024)
        self.backprops = 0
        self.task_id = 0
        self.history = {"task": [], "acc": [], "fg": [], "mem_bytes": [], "gens": [], "flops_fw": [], "backprops": []}

    def memory_bytes(self) -> int:
        total = sum(x.numel() * 4 for x in self.memory_f)
        total += sum(y.numel() for y in self.memory_y)
        return int(total)

    def _enforce_cap(self):
        while self.memory_bytes() > self.cap_bytes and len(self.memory_f) > 0:
            self.memory_f.pop(0)
            self.memory_y.pop(0)

    def sample(self, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.memory_f) == 0:
            return torch.empty(0, self.f_dim, device=self.cfg.device), torch.empty(0, dtype=torch.long, device=self.cfg.device)
        F = torch.cat(self.memory_f, dim=0)
        Y = torch.cat(self.memory_y, dim=0)
        N = F.size(0)
        idx = torch.randperm(N, device=F.device)[:min(K, N)]
        return F[idx], Y[idx]

    def train_task(self, train_loader: DataLoader, class_ids: List[int], warmup: bool = False):
        self.backprops = 0
        self.task_id += 1
        device = self.cfg.device
        self.backbone.train(); self.clf.train()
        if warmup and self.task_id == 1:
            for _ in range(self.cfg.epochs_warmup_task1):
                for x, y in train_loader:
                    if self.backprops >= self.cfg.C_backprops_per_task:
                        break
                    x, y = x.to(device), y.to(device)
                    f = self.backbone(x)
                    logits = self.clf(f)
                    loss = F.cross_entropy(logits, y)
                    self.opt.zero_grad(); loss.backward(); self.opt.step()
                    self.backprops += 1
                    if self.backprops >= self.cfg.C_backprops_per_task:
                        break
            self.backbone.freeze_low()
        for x, y in train_loader:
            if self.backprops >= self.cfg.C_backprops_per_task:
                break
            x, y = x.to(device), y.to(device)
            f = self.backbone(x)
            f_rep, y_rep = self.sample(self.cfg.replay_K)
            logits = self.clf(f)
            loss = F.cross_entropy(logits, y)
            if f_rep.numel() > 0:
                logits_rep = self.clf(f_rep)
                loss = loss + F.cross_entropy(logits_rep, y_rep)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.backprops += 1
            with torch.no_grad():
                take = min(x.size(0) // 2, 16)
                idx = torch.randperm(x.size(0), device=device)[:take]
                self.memory_f.append(f[idx].detach())
                self.memory_y.append(y[idx].detach())
                self._enforce_cap()

        self.history["task"].append(self.task_id)
        self.history["acc"].append(0.0)
        self.history["fg"].append(0.0)
        self.history["mem_bytes"].append(self.memory_bytes())
        self.history["gens"].append(0)
        self.history["flops_fw"].append(0.0)
        self.history["backprops"].append(self.backprops)
