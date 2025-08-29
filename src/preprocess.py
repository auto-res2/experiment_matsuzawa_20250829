# -*- coding: utf-8 -*-
"""
Data preprocessing for FREQUENT experiments.
Includes:
- PatternedDataset: simulates domain shifts via augment patterns.
- build_task_splits: splits dataset into tasks by classes.
- get_datasets: builds per-task Subset datasets for train/test.

Defaults use torchvision FakeData for quick tests to avoid downloads.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision as tv
import torchvision.transforms as T


class PatternedDataset(Dataset):
    """Wraps a base dataset with different augmentation patterns to simulate domain shifts."""
    def __init__(self, base: Dataset, pattern: str = 'standard'):
        self.base = base
        self.pattern = pattern
        # Define transforms (tensor-safe for modern torchvision)
        self.tr_standard = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
        ])
        self.tr_tinted = T.Compose([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.GaussianBlur(3, sigma=(0.1, 1.0)),
        ])
        self.tr_sketch = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.RandomAdjustSharpness(2.0),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if not isinstance(x, torch.Tensor):
            x = T.ToTensor()(x)
        if self.pattern == 'standard':
            x = self.tr_standard(x)
        elif self.pattern == 'tinted':
            x = self.tr_tinted(x)
        elif self.pattern == 'sketch':
            x = self.tr_sketch(x)
        return x, y


def build_task_splits(dataset: Dataset, n_tasks: int, classes_per_task: int,
                      subset_per_class: Optional[int] = None, seed: int = 0) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    labels = [int(dataset[idx][1]) for idx in range(len(dataset))]
    labels = np.array(labels)
    num_classes = int(labels.max()) + 1
    class_indices = [np.where(labels == c)[0].tolist() for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])
        if subset_per_class is not None:
            class_indices[c] = class_indices[c][:subset_per_class]
    assert n_tasks * classes_per_task <= num_classes, "Insufficient classes for tasks"
    tasks = []
    class_order = list(range(num_classes))
    rng.shuffle(class_order)
    for t in range(n_tasks):
        cls = class_order[t*classes_per_task:(t+1)*classes_per_task]
        idxs = []
        for c in cls:
            idxs += class_indices[c]
        rng.shuffle(idxs)
        tasks.append(idxs)
    return tasks


def get_datasets(cfg, pattern: str = 'standard') -> Tuple[List[Dataset], List[Dataset], int]:
    # Base dataset: FakeData by default (fast, no download)
    if getattr(cfg, 'use_fake_data', True):
        num_classes = cfg.n_tasks * cfg.classes_per_task
        base_train = tv.datasets.FakeData(size=num_classes*cfg.subset_per_class,
                                          image_size=(3, 32, 32), num_classes=num_classes,
                                          transform=T.ToTensor())
        base_test = tv.datasets.FakeData(size=max(1, num_classes*cfg.subset_per_class//2),
                                         image_size=(3, 32, 32), num_classes=num_classes,
                                         transform=T.ToTensor())
    else:
        name = getattr(cfg, 'dataset_name', 'CIFAR100').upper()
        if name == 'CIFAR100':
            base_train = tv.datasets.CIFAR100(root='./data', train=True, download=True, transform=T.ToTensor())
            base_test = tv.datasets.CIFAR100(root='./data', train=False, download=True, transform=T.ToTensor())
            num_classes = 100
        else:
            base_train = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
            base_test = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=T.ToTensor())
            num_classes = 10
    base_train = PatternedDataset(base_train, pattern=pattern)
    base_test = PatternedDataset(base_test, pattern='standard')
    tasks_train_idx = build_task_splits(base_train, cfg.n_tasks, cfg.classes_per_task, cfg.subset_per_class, seed=0)
    test_subset_per_class = cfg.subset_per_class//2 if getattr(cfg, 'subset_per_class', None) else None
    tasks_test_idx = build_task_splits(base_test, cfg.n_tasks, cfg.classes_per_task, test_subset_per_class, seed=1)
    train_tasks = [Subset(base_train, idxs) for idxs in tasks_train_idx]
    test_tasks = [Subset(base_test, idxs) for idxs in tasks_test_idx]
    return train_tasks, test_tasks, num_classes
