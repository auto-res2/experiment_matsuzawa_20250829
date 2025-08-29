# -*- coding: utf-8 -*-
"""
Main entry point for FREQUENT experiments.
Run from project root with:
    python -m src.main

It will execute a quick functionality test by default (FakeData, few steps)
and save figures into .research/iteration1/images.

Use --config config/config.yaml to customize.
"""
from __future__ import annotations
import os
import argparse
import yaml

from .train import (
    TrainConfig,
    set_seed,
    run_experiment1_strict,
    run_experiment2_ablation,
    run_experiment3_stream,
)


IMG_DIR = os.path.join('.research', 'iteration2', 'images')
os.makedirs(IMG_DIR, exist_ok=True)


def load_config(path: str) -> TrainConfig:
    with open(path, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    # Map YAML to TrainConfig with defaults
    cfg = TrainConfig(
        device=cfg_yaml.get('device', 'auto'),
        d=cfg_yaml.get('d', 64),
        M=cfg_yaml.get('M', 4),
        K=cfg_yaml.get('K', 256),
        freeze_depth=cfg_yaml.get('freeze_depth', 1),
        lr=cfg_yaml.get('lr', 0.05),
        momentum=cfg_yaml.get('momentum', 0.9),
        weight_decay=cfg_yaml.get('weight_decay', 1e-4),
        batch_size=cfg_yaml.get('batch_size', 32),
        epochs_per_task=cfg_yaml.get('epochs_per_task', 1),
        patience=cfg_yaml.get('patience', 1),
        replay_ratio_target_budget=cfg_yaml.get('replay_ratio_target_budget', 0.5),
        seed_list=cfg_yaml.get('seed_list', [123]),
        num_workers=cfg_yaml.get('num_workers', 0),
        mem_kb_list=cfg_yaml.get('mem_kb_list', [200]),
        n_tasks=cfg_yaml.get('n_tasks', 2),
        classes_per_task=cfg_yaml.get('classes_per_task', 5),
        use_fake_data=cfg_yaml.get('use_fake_data', True),
        dataset_name=cfg_yaml.get('dataset_name', 'CIFAR100'),
        subset_per_class=cfg_yaml.get('subset_per_class', 60),
        data_patterns=cfg_yaml.get('data_patterns', ['standard', 'tinted'])
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description='FREQUENT experimental runner')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to YAML config')
    parser.add_argument('--skip', type=str, default='', help='Comma-separated list of experiments to skip: exp1,exp2,exp3')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed_list[0] if len(cfg.seed_list) > 0 else 42)

    skip_set = set([s.strip().lower() for s in args.skip.split(',') if s.strip()])

    if 'exp1' not in skip_set:
        run_experiment1_strict(cfg, IMG_DIR)
    if 'exp2' not in skip_set:
        run_experiment2_ablation(cfg, IMG_DIR)
    if 'exp3' not in skip_set:
        run_experiment3_stream(cfg, IMG_DIR)

    # Report generated figures
    print("\n[MAIN] Completed. Generated PDF figures under:", IMG_DIR)
    try:
        for fn in sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.pdf')]):
            print(" -", os.path.join(IMG_DIR, fn))
    except Exception:
        pass


if __name__ == '__main__':
    main()
