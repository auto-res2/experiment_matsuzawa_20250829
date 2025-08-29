#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entrypoint to run ProPqEM experiments.
Usage: python -m src.main --config config/propqem_toy.yaml
This script performs three toy experiments and saves PDF figures to .research/iteration1/images.
"""
import os
import math
import json
from dataclasses import asdict
from typing import List, Optional
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
    _SEABORN_OK = True
except Exception:
    _SEABORN_OK = False

try:
    import pandas as pd  # type: ignore
    _PANDAS_OK = True
except Exception:
    _PANDAS_OK = False

try:
    from scipy import stats  # type: ignore
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

import yaml

from .train import TrainConfig, ProPqEMLearner, ERFeatureLearner
from .evaluate import evaluate_acc_propqem, evaluate_acc_er, average_forgetting, privacy_auc_loss_threshold
from .preprocess import build_fake_stream

IMAGES_DIR = os.path.join('.research', 'iteration1', 'images')


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_and_save(figpath: str):
    plt.tight_layout()
    plt.savefig(figpath, bbox_inches='tight', format='pdf')
    plt.close()


def run_experiment1(cfg: TrainConfig, num_tasks=3, classes_per_task=3, seeds: Optional[List[int]] = None):
    print("=== Experiment 1: ProPqEM vs ER-Feature (toy stream) ===")
    print("Config:", asdict(cfg))
    ensure_dir(IMAGES_DIR)

    tasks, total_classes = build_fake_stream(num_tasks=num_tasks, classes_per_task=classes_per_task)
    seeds = seeds if seeds is not None else [11, 22]

    all_hist_propqem = []
    all_hist_er = []

    for s in seeds:
        print(f"\n[Seed {s}] Run start...")
        set_seeds(s)
        propqem = ProPqEMLearner(num_classes=total_classes, cfg=cfg)
        er = ERFeatureLearner(num_classes=total_classes, cfg=cfg)

        from torch.utils.data import DataLoader
        t0_loader = DataLoader(tasks[0]["train"], batch_size=cfg.batch_size, shuffle=True)
        propqem.warm_start_pq(t0_loader, sketch_size=min(512, len(tasks[0]["train"])) )

        acc_matrix_propqem = []
        acc_matrix_er = []
        all_val_loaders = []

        for t, task in enumerate(tasks, start=1):
            train_loader = DataLoader(task["train"], batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(task["val"], batch_size=cfg.batch_size, shuffle=False)
            all_val_loaders.append(val_loader)

            propqem.train_task(train_loader, class_ids=task["class_ids"], warmup=True)
            er.train_task(train_loader, class_ids=task["class_ids"], warmup=True)

            # Evaluate on all seen tasks' validation sets
            accs_propqem = []
            accs_er = []
            for v in all_val_loaders:
                accs_propqem.append(evaluate_acc_propqem(propqem.backbone, propqem.clf, v, cfg.device))
                accs_er.append(evaluate_acc_er(er.backbone, er.clf, v, cfg.device))
            acc_matrix_propqem.append(accs_propqem)
            acc_matrix_er.append(accs_er)

        acc_matrix_propqem = np.array(acc_matrix_propqem)
        acc_matrix_er = np.array(acc_matrix_er)
        fgt_propqem = average_forgetting(acc_matrix_propqem)
        fgt_er = average_forgetting(acc_matrix_er)

        print(f"[Seed {s}] ProPqEM final ACC={acc_matrix_propqem[-1].mean():.3f} FGT={fgt_propqem:.3f} backprops_last={propqem.history['backprops'][-1]} mem={propqem.get_memory_bytes()/1e6:.3f}MB gens={len(propqem.pq.generations)}")
        print(f"[Seed {s}] ER-Feat final ACC={acc_matrix_er[-1].mean():.3f} FGT={fgt_er:.3f} backprops_last={er.history['backprops'][-1]} mem={er.memory_bytes()/1e6:.3f}MB")

        all_hist_propqem.append({"seed": s, "acc_matrix": acc_matrix_propqem.tolist(), "fg": fgt_propqem, "hist": propqem.history})
        all_hist_er.append({"seed": s, "acc_matrix": acc_matrix_er.tolist(), "fg": fgt_er, "hist": er.history, "mem_bytes": er.memory_bytes()})

    # Summary
    if _PANDAS_OK:
        rows = []
        for rec in all_hist_propqem:
            rows.append({"seed": rec["seed"], "final_acc": float(np.mean(rec["acc_matrix"][-1])), "fg": rec["fg"], "method": "ProPqEM"})
        for rec in all_hist_er:
            rows.append({"seed": rec["seed"], "final_acc": float(np.mean(rec["acc_matrix"][-1])), "fg": rec["fg"], "method": "ER-Feature"})
        df = pd.DataFrame(rows)
        print("\n[Exp1] Summary by seed:\n", df)
        try:
            df_p = df.pivot(index='seed', columns='method', values='final_acc')
            stats_result = None
            if _SCIPY_OK:
                t_stat, p_val = stats.ttest_rel(df_p['ProPqEM'], df_p['ER-Feature'])
                stats_result = {"t_stat": float(t_stat), "p_val": float(p_val)}
            print("[Exp1] Paired t-test ProPqEM vs ER-Feature (final_acc):", stats_result)
        except Exception as e:
            print("[Exp1] Stats aggregation error:", e)

    # Plots using the first seed's curves
    if len(all_hist_propqem) > 0:
        acc_curve_propqem = [np.mean(all_hist_propqem[0]["acc_matrix"][t]) for t in range(len(all_hist_propqem[0]["acc_matrix"]))]
        acc_curve_er = [np.mean(all_hist_er[0]["acc_matrix"][t]) for t in range(len(all_hist_er[0]["acc_matrix"]))]
        plt.figure(figsize=(5, 3))
        if _SEABORN_OK:
            sns.lineplot(x=list(range(1, len(acc_curve_propqem)+1)), y=acc_curve_propqem, label='ProPqEM')
            sns.lineplot(x=list(range(1, len(acc_curve_er)+1)), y=acc_curve_er, label='ER-Feature')
        else:
            plt.plot(range(1, len(acc_curve_propqem)+1), acc_curve_propqem, label='ProPqEM')
            plt.plot(range(1, len(acc_curve_er)+1), acc_curve_er, label='ER-Feature')
        plt.xlabel('Task'); plt.ylabel('Average Accuracy'); plt.title('ACC over tasks (toy)'); plt.legend()
        plot_and_save(os.path.join(IMAGES_DIR, 'accuracy_propqem_vs_er.pdf'))

        mem_bytes = all_hist_propqem[0]["hist"]["mem_bytes"]
        plt.figure(figsize=(5, 3))
        plt.plot(range(1, len(mem_bytes)+1), [mb/1e6 for mb in mem_bytes], marker='o')
        plt.xlabel('Task'); plt.ylabel('Memory (MB)'); plt.title('Memory vs Tasks (ProPqEM)')
        plot_and_save(os.path.join(IMAGES_DIR, 'memory_vs_tasks_propqem.pdf'))

        pq_mse = all_hist_propqem[0]["hist"]["pq_mse"]
        plt.figure(figsize=(5, 3))
        plt.plot(range(1, len(pq_mse)+1), pq_mse, marker='o')
        plt.xlabel('Task'); plt.ylabel('PQ MSE (moving)'); plt.title('PQ Recon Error by Task (ProPqEM)')
        plot_and_save(os.path.join(IMAGES_DIR, 'pq_recon_error_propqem.pdf'))

    print("[Exp1] Figures saved in:", IMAGES_DIR)
    return all_hist_propqem, all_hist_er


def run_experiment2(cfg: TrainConfig, seeds: Optional[List[int]] = None):
    print("=== Experiment 2: Ablations & long-horizon diagnostics (toy) ===")
    ensure_dir(IMAGES_DIR)

    tasks, total_classes = build_fake_stream(num_tasks=10, classes_per_task=2, samples_per_class=60)
    seeds = seeds if seeds is not None else [33, 44]

    slopes = []
    bc_drops = []

    from torch.utils.data import DataLoader

    for s in seeds[:2]:
        print(f"\n[Seed {s}] Long-horizon toy run...")
        set_seeds(s)
        learner = ProPqEMLearner(num_classes=total_classes, cfg=cfg)
        t0_loader = DataLoader(tasks[0]["train"], batch_size=cfg.batch_size, shuffle=True)
        learner.warm_start_pq(t0_loader, sketch_size=min(512, len(tasks[0]["train"])) )

        items_per_class_log = {}
        prototypes_per_class = {}
        task1_codes = None
        task1_gens = None
        task1_feats = None
        all_val_loaders = []

        for t, task in enumerate(tasks, start=1):
            train_loader = DataLoader(task["train"], batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(task["val"], batch_size=cfg.batch_size, shuffle=False)
            all_val_loaders.append(val_loader)

            learner.train_task(train_loader, class_ids=task["class_ids"], warmup=True)

            codes, labels, gens = learner.memory.as_tensors()
            for c in labels.unique().tolist():
                Nc = int((labels == c).sum().item())
                items_per_class_log[c] = items_per_class_log.get(c, 0) + Nc
                prototypes_per_class[c] = Nc

            if t == 1:
                task1_codes, _, task1_gens = learner.memory.as_tensors()
                with torch.no_grad():
                    if task1_codes.numel() > 0:
                        z1 = learner.pq.decode(task1_codes, task1_gens)
                        task1_feats = learner.delta_dec(z1)

        Ns = np.array([max(1, items_per_class_log[c]) for c in sorted(items_per_class_log.keys())])
        Ks = np.array([prototypes_per_class[c] for c in sorted(prototypes_per_class.keys())])
        x = np.log(Ns + 1e-8); y = Ks.astype(float)
        A = np.vstack([np.ones_like(x), x]).T
        coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        slope = float(coeff[1]) if coeff.shape[0] > 1 else float('nan')
        slopes.append(slope)
        print(f"[Seed {s}] Sub-linear trend slope (prototypes vs log N): {slope:.3f}")

        bc_drop = float('nan')
        if task1_codes is not None and task1_codes.numel() > 0:
            latest_gen_id = len(learner.pq.generations) - 1
            with torch.no_grad():
                zT = learner.pq.decode(task1_codes, torch.full_like(task1_gens, latest_gen_id))
                fT = learner.delta_dec(zT)
            with torch.no_grad():
                logits1 = learner.clf(task1_feats.to(cfg.device))
                logitsT = learner.clf(fT.to(cfg.device))
                pred1 = logits1.argmax(dim=1)
                predT = logitsT.argmax(dim=1)
                acc1 = (pred1 == pred1).float().mean().item()
                accT = (predT == pred1).float().mean().item()
                bc_drop = max(0.0, acc1 - accT)
        bc_drops.append(bc_drop)
        print(f"[Seed {s}] Backward-compat drop (earliest codes decoded with latest): {bc_drop:.4f}")

    plt.figure(figsize=(5, 3))
    plt.hist(slopes, bins=5)
    plt.xlabel('Slope (prototypes vs log N)'); plt.ylabel('Count'); plt.title('Sub-linear memory slope (toy)')
    plot_and_save(os.path.join(IMAGES_DIR, 'prototypes_log_slope.pdf'))

    plt.figure(figsize=(5, 3))
    plt.hist(bc_drops, bins=5)
    plt.xlabel('Backward-compat ACC drop'); plt.ylabel('Count'); plt.title('Backward Compatibility (toy)')
    plot_and_save(os.path.join(IMAGES_DIR, 'backward_compat_drop_propqem.pdf'))

    print("[Exp2] Figures saved in:", IMAGES_DIR)
    return slopes, bc_drops


def run_experiment3(cfg: TrainConfig, seeds: Optional[List[int]] = None):
    print("=== Experiment 3: Privacy evaluation aligned with stored artifacts (toy) ===")
    ensure_dir(IMAGES_DIR)

    tasks, total_classes = build_fake_stream(num_tasks=4, classes_per_task=5, samples_per_class=60)
    seeds = seeds if seeds is not None else [55, 66]

    aucs_propqem = []
    aucs_er = []

    from torch.utils.data import DataLoader

    for s in seeds[:2]:
        print(f"\n[Seed {s}] Training for privacy artifact extraction...")
        set_seeds(s)
        learner = ProPqEMLearner(num_classes=total_classes, cfg=cfg)
        er = ERFeatureLearner(num_classes=total_classes, cfg=cfg)
        t0_loader = DataLoader(tasks[0]["train"], batch_size=cfg.batch_size, shuffle=True)
        learner.warm_start_pq(t0_loader, sketch_size=min(512, len(tasks[0]["train"])) )
        for t, task in enumerate(tasks, start=1):
            train_loader = DataLoader(task["train"], batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(task["val"], batch_size=cfg.batch_size, shuffle=False)
            learner.train_task(train_loader, class_ids=task["class_ids"], warmup=True)
            er.train_task(train_loader, class_ids=task["class_ids"], warmup=True)

        codes, labels, gens = learner.memory.as_tensors()
        if codes.numel() == 0:
            continue
        with torch.no_grad():
            z_hat = learner.pq.decode(codes, gens)
            f_hat = learner.delta_dec(z_hat)

        val_loader = DataLoader(tasks[-1]["val"], batch_size=cfg.batch_size, shuffle=True)
        f_non = []
        y_lab = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(cfg.device)
                f = learner.backbone(x)
                f_non.append(f.detach().cpu())
                y_lab.append(y)
                if len(torch.cat(f_non)) >= min(f_hat.size(0), 256):
                    break
        if len(f_non) == 0:
            continue
        f_non = torch.cat(f_non)[:min(f_hat.size(0), 256)]
        y_lab = torch.cat(y_lab)[:f_non.size(0)]
        f_mem = f_hat[:f_non.size(0)].detach().cpu()
        auc_propqem = privacy_auc_loss_threshold(learner.clf, f_mem, f_non, y_lab, cfg.device)
        aucs_propqem.append(auc_propqem)
        print(f"[Seed {s}] Privacy AUC (ProPqEM codes->f): {auc_propqem:.3f}")

        F = torch.cat(er.memory_f, dim=0) if len(er.memory_f) > 0 else torch.empty(0, cfg.f_dim)
        Y = torch.cat(er.memory_y, dim=0) if len(er.memory_y) > 0 else torch.empty(0, dtype=torch.long)
        if F.numel() > 0:
            f_mem2 = F[:f_non.size(0)].detach().cpu()
            y_lab2 = Y[:f_non.size(0)].detach().cpu()
            auc_er = privacy_auc_loss_threshold(er.clf, f_mem2, f_non, y_lab2, cfg.device)
            aucs_er.append(auc_er)
            print(f"[Seed {s}] Privacy AUC (ER-Feature raw f): {auc_er:.3f}")

    plt.figure(figsize=(5, 3))
    bins = np.linspace(0.0, 1.0, 11)
    plt.hist(aucs_propqem, bins=bins, alpha=0.7, label='ProPqEM')
    if len(aucs_er) > 0:
        plt.hist(aucs_er, bins=bins, alpha=0.7, label='ER-Feature')
    plt.xlabel('MI AUC'); plt.ylabel('Count'); plt.title('Privacy AUC distribution (toy)'); plt.legend()
    plot_and_save(os.path.join(IMAGES_DIR, 'privacy_auc_propqem.pdf'))

    print("[Exp3] Figure saved in:", IMAGES_DIR)
    return aucs_propqem, aucs_er


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/propqem_toy.yaml', help='Path to YAML config')
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg_raw = yaml.safe_load(f)
    else:
        print(f"[Warning] Config {args.config} not found. Using defaults.")
        cfg_raw = {}

    cfg = TrainConfig(**{k: v for k, v in cfg_raw.get('train_config', {}).items()}) if 'train_config' in cfg_raw else TrainConfig()

    # Quick smoke tests on toy streams
    hist_p, hist_e = run_experiment1(cfg, num_tasks=cfg_raw.get('exp1', {}).get('num_tasks', 3),
                                     classes_per_task=cfg_raw.get('exp1', {}).get('classes_per_task', 3),
                                     seeds=cfg_raw.get('exp1', {}).get('seeds', [11, 22]))
    slopes, bc = run_experiment2(cfg, seeds=cfg_raw.get('exp2', {}).get('seeds', [33, 44]))
    auc_p, auc_e = run_experiment3(cfg, seeds=cfg_raw.get('exp3', {}).get('seeds', [55, 66]))

    print("\n=== Quick Test Summary ===")
    if len(hist_p) > 0:
        print("ProPqEM final ACC (seed1):", np.mean(hist_p[0]["acc_matrix"][-1]))
    if len(hist_e) > 0:
        print("ER-Feature final ACC (seed1):", np.mean(hist_e[0]["acc_matrix"][-1]))
    print("Sub-linear slopes (toy):", slopes)
    print("Backward-compat drops (toy):", bc)
    print("Privacy AUC ProPqEM (toy):", auc_p)


if __name__ == '__main__':
    main()
