import os
import torch
import numpy as np
from torchvision import datasets, transforms


def build_torchvision_loader(args, is_train=False):
    t = []
    input_size = 224
    crop_pct = 224 / 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size)
    )
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    tforms = transforms.Compose(t)
    root = os.path.join(args.data_path, "train" if is_train else "val")
    dataset = datasets.ImageFolder(root, transform=tforms)
    sampler = None
    if hasattr(args, 'subset_len'):
        if args.subset_len is not None:
            total_images = len(dataset)
            image_indices = list(range(total_images))
            np.random.shuffle(image_indices)
            sampler = torch.utils.data.SubsetRandomSampler(image_indices[:args.subset_len])
    valid_queue = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.val_batch_size),
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=False,
        drop_last=False,
    )
    return valid_queue, dataset


def build_torchvision_loader_tpu(args):
    import torch_xla.core.xla_model as xm
    t = []
    input_size = 224
    crop_pct = 224 / 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size)
    )
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    tforms = transforms.Compose(t)
    root = os.path.join(args.data_path, "train")
    train_dataset = datasets.ImageFolder(root, transform=tforms)
    root = os.path.join(args.data_path, "val")
    test_dataset = datasets.ImageFolder(root, transform=tforms)
    train_sampler, test_sampler = None, None
    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
          test_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        sampler=test_sampler,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers)

    return train_loader, test_loader
