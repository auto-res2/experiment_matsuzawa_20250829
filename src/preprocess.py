#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing and data stream builders for toy continual learning experiments.
"""
from typing import List, Tuple
from torchvision import datasets, transforms
from torch.utils.data import Subset


def build_fake_stream(num_tasks: int = 4,
                      classes_per_task: int = 5,
                      samples_per_class: int = 80,
                      image_size=(3, 32, 32)) -> Tuple[List[dict], int]:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    total_classes = num_tasks * classes_per_task
    dataset = datasets.FakeData(size=total_classes * samples_per_class,
                                image_size=image_size,
                                num_classes=total_classes,
                                transform=transform)
    class_order = list(range(total_classes))
    tasks: List[dict] = []
    # Build task splits with disjoint classes
    # We split by scanning the dataset and selecting items with labels in cls_ids
    for t in range(num_tasks):
        cls_start = t * classes_per_task
        cls_ids = class_order[cls_start: cls_start + classes_per_task]
        idxs = []
        for i in range(len(dataset)):
            _, y = dataset[i]
            if y in cls_ids:
                idxs.append(i)
        n = len(idxs)
        n_val = max(1, n // 5)
        val_idx = idxs[:n_val]
        train_idx = idxs[n_val:]
        tasks.append({
            "train": Subset(dataset, train_idx),
            "val": Subset(dataset, val_idx),
            "class_ids": cls_ids,
        })
    return tasks, total_classes
