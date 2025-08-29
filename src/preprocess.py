# -*- coding: utf-8 -*-
import os
import json
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_split_cifar100(root: str, seed: int, tasks: int, classes_per_task: int, train: bool,
                         transform, limit_classes=None, subset_per_class=None):
    set_seed(seed)
    full = torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    all_classes = list(range(100))
    rng = np.random.RandomState(seed)
    rng.shuffle(all_classes)
    if limit_classes is not None:
        all_classes = all_classes[:limit_classes]
        tasks = max(1, limit_classes // classes_per_task)

    task_classes = [sorted(all_classes[i*classes_per_task:(i+1)*classes_per_task]) for i in range(tasks)]
    per_task_indices = []
    labels_np = np.array(full.targets)

    for cls_group in task_classes:
        idxs = np.where(np.isin(labels_np, cls_group))[0]
        if subset_per_class is not None:
            selected = []
            for c in cls_group:
                ci = np.where(labels_np == c)[0]
                rng.shuffle(ci)
                ci = ci[:subset_per_class]
                selected.append(ci)
            idxs = np.concatenate(selected)
        per_task_indices.append(np.array(sorted(idxs)))

    datasets = [Subset(full, idxs.tolist()) for idxs in per_task_indices]
    return datasets, task_classes


def make_class_remap(task_classes: List[List[int]]):
    cumulative_map = {}
    next_label = 0
    remaps = []
    for cls_group in task_classes:
        m = {}
        for c in cls_group:
            if c not in cumulative_map:
                cumulative_map[c] = next_label
                next_label += 1
            m[c] = cumulative_map[c]
        remaps.append(m)
    return remaps, cumulative_map


class RemapTargets(torch.utils.data.Dataset):
    def __init__(self, subset: Subset, mapping: dict):
        self.subset = subset
        self.mapping = mapping
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.mapping[y]


def build_loaders(datasets, mapping_list, batch_size=64, num_workers=2, shuffle=True):
    loaders = []
    for ds, mp in zip(datasets, mapping_list):
        ds_mapped = RemapTargets(ds, mp)
        loaders.append(DataLoader(ds_mapped, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers))
    return loaders
