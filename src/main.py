# -*- coding: utf-8 -*-
import os
import sys
import json
from typing import Tuple

import yaml
import numpy as np
import torch

from .preprocess import set_seed, build_split_cifar100, make_class_remap, build_loaders
from .train import train_continual
from .evaluate import confusion_matrix, save_confusion_matrix_pdf


def main():
    # Load config
    cfg_path = os.environ.get('CL_CONFIG', 'config/default.yaml')
    if not os.path.exists(cfg_path):
        print(f"Config file not found at {cfg_path}. Please create it.")
        sys.exit(1)
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get('exp_name', 'iteration4_frequent')
    dataset_root = cfg.get('dataset_root', 'data')
    seeds = tuple(cfg.get('seeds', [0]))
    budgets = tuple(cfg.get('budgets', ['50kB']))
    compute_alphas = tuple(cfg.get('compute_alphas', [0.5]))
    methods = tuple(cfg.get('methods', ['FREQUENT','ER']))
    epochs_per_task = int(cfg.get('epochs_per_task', 1))
    tasks = int(cfg.get('tasks', 2))
    classes_per_task = int(cfg.get('classes_per_task', 5))
    batch_size = int(cfg.get('batch_size', 32))
    quick_test = bool(cfg.get('quick_test', True))
    num_workers = int(cfg.get('num_workers', 2))
    image_dir = cfg.get('save_dir_images', '.research/iteration4/images')

    os.makedirs(image_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Transforms
    import torchvision.transforms as T
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    transform_test = T.Compose([T.ToTensor()])

    for budget_tag in budgets:
        for alpha in compute_alphas:
            all_seed_results = {}
            for seed in seeds:
                print(f"[Run] exp={exp_name} budget={budget_tag} alpha={alpha} seed={seed} methods={methods}")
                set_seed(seed)
                # Data splits (quick test uses fewer classes and subset)
                limit_classes = 20 if quick_test else None
                subset_per_class = 50 if quick_test else None

                train_tasks, task_classes = build_split_cifar100(
                    dataset_root, seed=seed, tasks=tasks, classes_per_task=classes_per_task,
                    train=True, transform=transform_train, limit_classes=limit_classes,
                    subset_per_class=subset_per_class
                )
                test_tasks, _ = build_split_cifar100(
                    dataset_root, seed=seed, tasks=tasks, classes_per_task=classes_per_task,
                    train=False, transform=transform_test, limit_classes=limit_classes,
                    subset_per_class=None
                )
                remaps, cum_map = make_class_remap(task_classes)
                num_classes_total = len(cum_map)

                train_loaders = build_loaders(train_tasks, remaps, batch_size=batch_size, num_workers=num_workers, shuffle=True)
                test_loaders = build_loaders(test_tasks, remaps, batch_size=256, num_workers=num_workers, shuffle=False)

                results = train_continual(
                    exp_name=exp_name,
                    out_image_dir=image_dir,
                    train_loaders=train_loaders,
                    test_loaders=test_loaders,
                    num_classes_total=num_classes_total,
                    methods=methods,
                    budget_tag=budget_tag,
                    alpha=alpha,
                    seed=seed,
                    epochs_per_task=epochs_per_task,
                    classes_per_task=classes_per_task,
                    batch_size=batch_size,
                    quick_test=quick_test,
                )

                # Save per-seed JSON summary
                os.makedirs('results', exist_ok=True)
                out_json = os.path.join('results', f"{exp_name}_{budget_tag}_a{alpha}_seed{seed}.json")
                with open(out_json, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved results -> {out_json}")

                all_seed_results[seed] = results

            # Aggregate across seeds and print summary
            for method in methods:
                accs = []
                forg = []
                for seed in seeds:
                    accs.append(all_seed_results[seed]['per_method'][method]['final_avg_acc'])
                    forg.append(all_seed_results[seed]['per_method'][method]['forgetting'])
                mean_acc = float(np.mean(accs))
                std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
                mean_f = float(np.mean(forg))
                std_f = float(np.std(forg, ddof=1)) if len(forg) > 1 else 0.0
                print(f"[Summary] {method} {budget_tag} alpha={alpha}  acc={mean_acc:.4f}±{std_acc:.4f}  forgetting={mean_f:.4f}±{std_f:.4f}")

    print("Experiment finished. All figures saved as PDF in:", image_dir)


if __name__ == '__main__':
    main()
