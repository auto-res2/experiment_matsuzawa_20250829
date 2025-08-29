"""
Data preprocessing and task stream construction for continual learning experiments.
"""
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms


def make_transforms(img_size=32):
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return tf_train, tf_test


def get_dataset(name: str, train: bool, transform):
    name = name.lower()
    if name == 'cifar100':
        return torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    elif name == 'cifar10':
        return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif name == 'fake':
        size = 2000 if train else 500
        return torchvision.datasets.FakeData(size=size, image_size=(3,32,32), num_classes=10, transform=transform)
    else:
        return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)


class TaskSubset(Dataset):
    def __init__(self, base: Dataset, class_ids: List[int], limit_per_class: Optional[int] = None, seed: int = 0):
        super().__init__()
        self.base = base
        self.class_ids = set(class_ids)
        self.indices = []
        rng = np.random.RandomState(seed)
        per_class_count = {c: 0 for c in class_ids}
        for idx in range(len(base)):
            _, y = base[idx]
            if int(y) in self.class_ids:
                if limit_per_class is None or per_class_count[int(y)] < limit_per_class:
                    self.indices.append(idx)
                    if limit_per_class is not None:
                        per_class_count[int(y)] += 1
        rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def build_task_stream(dataset_name: str, num_tasks: int, classes_per_task: int, seed: int = 0,
                      limit_per_class: Optional[int] = None, img_size: int = 32):
    tf_train, tf_test = make_transforms(img_size)
    train_base = get_dataset(dataset_name, True, tf_train)
    test_base = get_dataset(dataset_name, False, tf_test)

    if hasattr(train_base, 'classes'):
        num_classes = len(train_base.classes)
    elif hasattr(train_base, 'targets'):
        num_classes = int(max(train_base.targets)) + 1
    else:
        num_classes = 10
    classes = list(range(num_classes))
    rng = np.random.RandomState(seed)
    rng.shuffle(classes)

    tasks = []
    for t in range(num_tasks):
        cls = classes[t*classes_per_task:(t+1)*classes_per_task]
        if len(cls) == 0:
            break
        train_ds = TaskSubset(train_base, cls, limit_per_class=limit_per_class, seed=seed+100*t)
        test_ds = TaskSubset(test_base, cls, limit_per_class=None, seed=seed+200*t)
        val_indices = list(range(0, max(1, len(test_ds)//5)))
        val_ds = Subset(test_ds, val_indices)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)
        tasks.append((t, train_loader, val_loader, test_loader, cls))
    return tasks, num_classes
