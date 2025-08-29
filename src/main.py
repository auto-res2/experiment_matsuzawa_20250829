"""
Main entry point. Run experiments with:
  python -m src.main --config config/config.yaml
Default runs a quick smoke test on FakeData and saves PDFs under .research/iteration2/images.
"""
import os
import argparse
import yaml
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .preprocess import build_task_stream
from .train import TrainConfig, train_propqem, train_baseline_raw_feature
from .evaluate import roc_auc_from_scores


def run_smoke(cfg: dict):
    device = torch.device(cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    save_dir = cfg.get('save_dir', '.research/iteration2/images')
    os.makedirs(save_dir, exist_ok=True)

    dataset_name = cfg.get('dataset_name', 'fake')
    num_tasks = int(cfg.get('num_tasks', 2))
    classes_per_task = int(cfg.get('classes_per_task', 5))
    limit_per_class = cfg.get('limit_per_class', 50)
    seed = int(cfg.get('seed', 42))

    stream, num_classes = build_task_stream(dataset_name, num_tasks, classes_per_task, seed=seed, limit_per_class=limit_per_class)

    print(f"Smoke run: dataset={dataset_name}, tasks={num_tasks}, classes/task={classes_per_task}, device={device}")

    tc_prop = cfg.get('train_config_propqem', {})
    prop_cfg = TrainConfig(
        lr=float(tc_prop.get('lr', 0.05)),
        weight_decay=float(tc_prop.get('weight_decay', 5e-4)),
        momentum=float(tc_prop.get('momentum', 0.9)),
        epochs_per_task=int(tc_prop.get('epochs_per_task', 1)),
        warmup_epochs_first_task=int(tc_prop.get('warmup_epochs_first_task', 1)),
        freeze_after_warmup=bool(tc_prop.get('freeze_after_warmup', True)),
        tau_recon=float(tc_prop.get('tau_recon', 0.2)),
        rho_gis=float(tc_prop.get('rho_gis', 1.0)),
        replay_K=int(tc_prop.get('replay_K', 16)),
        budget_C=int(tc_prop.get('budget_C', 80)),
        mem_cap_mb=float(tc_prop.get('mem_cap_mb', 1.0)),
    )

    stats_prop = train_propqem(stream, num_classes, prop_cfg, device, seed=seed, save_dir=save_dir, verbose=True)

    tc_base = cfg.get('train_config_baseline', {})
    base_cfg = TrainConfig(
        lr=float(tc_base.get('lr', 0.05)),
        weight_decay=float(tc_base.get('weight_decay', 5e-4)),
        momentum=float(tc_base.get('momentum', 0.9)),
        epochs_per_task=int(tc_base.get('epochs_per_task', 1)),
        warmup_epochs_first_task=0,
        replay_K=int(tc_base.get('replay_K', 16)),
        mem_cap_mb=float(tc_base.get('mem_cap_mb', 1.0)),
    )
    stats_base = train_baseline_raw_feature(stream, num_classes, base_cfg, device, seed=seed, save_dir=save_dir, verbose=True)

    print("Smoke test complete. Figures saved under:", os.path.abspath(save_dir))


def run_experiment1(cfg: dict):
    device = torch.device(cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    save_dir = cfg.get('save_dir', '.research/iteration2/images')
    os.makedirs(save_dir, exist_ok=True)

    dataset_name = cfg.get('dataset_name', 'cifar100')
    num_tasks = int(cfg.get('num_tasks', 4))
    classes_per_task = int(cfg.get('classes_per_task', 5))
    limit_per_class = cfg.get('limit_per_class', 100)
    seeds = cfg.get('seeds', [0])
    mem_caps_mb = cfg.get('mem_caps_mb', [1.0, 5.0, 20.0])

    stream, num_classes = build_task_stream(dataset_name, num_tasks, classes_per_task, seed=seeds[0], limit_per_class=limit_per_class)

    results = {}
    for mem_cap in mem_caps_mb:
        print(f"\n=== Memory cap: {mem_cap} MB ===")
        cfg_prop = TrainConfig(epochs_per_task=1, warmup_epochs_first_task=1, tau_recon=0.12, rho_gis=1.0,
                               replay_K=32, budget_C=200, mem_cap_mb=float(mem_cap), lr=0.05)
        stats_prop = train_propqem(stream, num_classes, cfg_prop, device, seed=seeds[0], save_dir=save_dir, verbose=True)

        cfg_base = TrainConfig(epochs_per_task=1, replay_K=32, mem_cap_mb=float(mem_cap), lr=0.05)
        stats_base = train_baseline_raw_feature(stream, num_classes, cfg_base, device, seed=seeds[0], save_dir=save_dir, verbose=True)

        results[mem_cap] = {
            'propqem': stats_prop.acc_per_task[-1] if stats_prop.acc_per_task else 0.0,
            'baseline': stats_base.acc_per_task[-1] if stats_base.acc_per_task else 0.0,
        }
        print(f"[Summary @ {mem_cap} MB] ProPqEM ACC={results[mem_cap]['propqem']*100:.2f}% | Baseline ACC={results[mem_cap]['baseline']*100:.2f}%")

    caps = sorted(results.keys())
    accs_prop = [results[c]['propqem'] for c in caps]
    accs_base = [results[c]['baseline'] for c in caps]

    plt.figure(figsize=(5,3))
    plt.plot(caps, accs_prop, marker='o', label='ProPqEM')
    plt.plot(caps, accs_base, marker='s', label='Baseline-ER (raw f)')
    plt.xlabel('Memory cap (MB)')
    plt.ylabel('Last ACC')
    plt.title('ACC vs Memory Cap (Exp1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_memcap_comparison.pdf'), bbox_inches='tight')
    plt.close()

    print("Experiment 1 completed. Figures saved under:", os.path.abspath(save_dir))


def run_experiment2(cfg: dict):
    device = torch.device(cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    save_dir = cfg.get('save_dir', '.research/iteration2/images')
    os.makedirs(save_dir, exist_ok=True)

    dataset_name = cfg.get('dataset_name', 'cifar100')
    num_tasks = int(cfg.get('num_tasks', 10))
    classes_per_task = int(cfg.get('classes_per_task', 1))
    limit_per_class = cfg.get('limit_per_class', 50)
    seed = int(cfg.get('seed', 0))

    stream, num_classes = build_task_stream(dataset_name, num_tasks, classes_per_task, seed=seed, limit_per_class=limit_per_class)

    cfg_prop = TrainConfig(epochs_per_task=1, warmup_epochs_first_task=1, tau_recon=0.1, rho_gis=1.0,
                           replay_K=16, budget_C=150, mem_cap_mb=50.0, lr=0.05)
    stats = train_propqem(stream, num_classes, cfg_prop, device, seed=seed, save_dir=save_dir, verbose=True)

    plt.figure(figsize=(5,3))
    plt.plot(stats.memory_mb_per_task, marker='o')
    plt.xlabel('Task')
    plt.ylabel('Memory (MB)')
    plt.title('Memory vs Tasks (Exp2)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'memory_vs_tasks_propqem.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(5,3))
    plt.plot(stats.gen_count_per_task, marker='s')
    plt.xlabel('Task')
    plt.ylabel('# PQ Generations')
    plt.title('Codebook Generations vs Tasks (Exp2)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'generations_vs_tasks_propqem.pdf'), bbox_inches='tight')
    plt.close()

    print("Experiment 2 completed. Figures saved under:", os.path.abspath(save_dir))


def run_experiment3(cfg: dict):
    # Privacy and compute-budget sensitivity (lightweight version)
    device = torch.device(cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    save_dir = cfg.get('save_dir', '.research/iteration2/images')
    os.makedirs(save_dir, exist_ok=True)

    dataset_name = cfg.get('dataset_name', 'cifar100')
    num_tasks = int(cfg.get('num_tasks', 4))
    classes_per_task = int(cfg.get('classes_per_task', 5))
    limit_per_class = cfg.get('limit_per_class', 100)
    seed = int(cfg.get('seed', 0))

    stream, num_classes = build_task_stream(dataset_name, num_tasks, classes_per_task, seed=seed, limit_per_class=limit_per_class)

    # ProPqEM
    cfg_prop = TrainConfig(epochs_per_task=1, warmup_epochs_first_task=1, replay_K=32, budget_C=200, mem_cap_mb=5.0, lr=0.05)
    stats_prop = train_propqem(stream, num_classes, cfg_prop, device, seed=seed, save_dir=save_dir, verbose=True)

    # Baseline
    cfg_raw = TrainConfig(epochs_per_task=1, replay_K=32, mem_cap_mb=5.0, lr=0.05)
    stats_raw = train_baseline_raw_feature(stream, num_classes, cfg_raw, device, seed=seed, save_dir=save_dir, verbose=True)

    print("Experiment 3 minimal run complete. Figures saved under:", os.path.abspath(save_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to YAML config')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        # Fallback to defaults if config file missing
        cfg = {}
    else:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

    mode = cfg.get('mode', 'smoke').lower()

    if mode == 'smoke':
        run_smoke(cfg)
    elif mode == 'exp1':
        run_experiment1(cfg)
    elif mode == 'exp2':
        run_experiment2(cfg)
    elif mode == 'exp3':
        run_experiment3(cfg)
    else:
        print(f"Unknown mode '{mode}'. Supported: smoke|exp1|exp2|exp3")


if __name__ == '__main__':
    main()
