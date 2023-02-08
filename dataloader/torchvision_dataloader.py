import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
from timm.data import constants
from timm.data import create_dataset, create_loader 


def build_torchvision_loader(args, is_train=False):
    t = []
    input_size = 224
    crop_pct = 224 / 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    size = int(input_size / crop_pct)
    t.append(transforms.Resize(size))
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    tforms = transforms.Compose(t)
    root = os.path.join(args.data_path, "train" if is_train else "val")
    dataset = datasets.ImageFolder(root, transform=tforms)
    sampler = None
    if hasattr(args, "subset_len"):
        if args.subset_len is not None:
            total_images = len(dataset)
            image_indices = list(range(total_images))
            np.random.shuffle(image_indices)
            sampler = torch.utils.data.SubsetRandomSampler(
                image_indices[: args.subset_len]
            )
    valid_queue = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.val_batch_size),
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=False,
        drop_last=False,
    )
    return valid_queue, dataset


def build_loader_timm(args):
    args.input_size = 224
    args.imagenet_default_mean_and_std = True
    args.train_interpolation = "bicubic"
    args.reprob = 0.20
    args.remode = "pixel"
    args.recount = 1
    args.resplit = False
    # args.aa = "rand-m9-mstd0.5-inc1"
    args.aa = "rand-m9-mstd0.5"
    args.scale = [0.08, 1.0]
    args.ratio = [3. / 4., 4. / 3.]
    args.color_jitter = 0.4
    args.hflip = 0.5
    args.vflip = 0.0
    args.color_jitter = 0.4
    args.aug_repeats = 0
    num_aug_splits = 0
    train_interpolation = 'random'
    collate_fn = None
    args.prefetcher=True
    dataset_train = create_dataset(
        'imagefolder',
        root=args.train_dataset,
        split='train',
        is_training=True,
        class_map='',
        download=False,
        batch_size=args.train_batch_size,
        seed=args.seed,
        repeats=0,
    )
    dataset_eval = create_dataset(
        'imagefolder',
        root=args.train_dataset,
        split='validation',
        is_training=False,
        class_map='',
        download=False,
        batch_size=args.val_batch_size,
    )
    loader_train = create_loader(
        dataset_train,
        input_size=(3,224,224),
        batch_size=args.train_batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=False,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=constants.IMAGENET_DEFAULT_MEAN,
        std=constants.IMAGENET_DEFAULT_STD,
        num_workers=args.num_workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=False,
        device=torch.device(f'cuda:{args.local_rank}'),
        use_multi_epochs_loader=False,
        worker_seeding='all',
    )
    loader_eval = create_loader(
        dataset_eval,
        input_size=(3,224,224),
        batch_size=None or args.train_batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation='bicubic',
        mean=constants.IMAGENET_DEFAULT_MEAN,
        std=constants.IMAGENET_DEFAULT_STD,
        num_workers=args.num_workers,
        distributed=args.distributed,
        crop_pct=constants.DEFAULT_CROP_PCT,
        pin_memory=False,
        device=torch.device(f'cuda:{args.local_rank}'),
    )
    return loader_train, loader_eval, dataset_train 


def build_torchvision_loader_tpu(args):
    import torch_xla.core.xla_model as xm
    from dataloader.datasets import build_transform

    args.input_size = 224
    args.imagenet_default_mean_and_std = True
    args.train_interpolation = "bicubic"
    args.reprob = 0.20
    args.remode = "pixel"
    args.recount = 1
    args.aa = "rand-m9-mstd0.5-inc1"
    # args.aa = "rand-m9-mstd0.5"
    args.color_jitter = 0.4
    args.crop_pct = None  # 224 / 256

    train_transforms = build_transform(is_train=True, args=args)
    val_transforms = build_transform(is_train=False, args=args)

    '''
    print("---" * 10)
    print("Train Transforms:")
    for t in train_transforms.transforms:
        print(t)
    print("---" * 10)
    print("---" * 10)
    print("Val Transforms:")
    for t in val_transforms.transforms:
        print(t)
    print("---" * 10)
    '''

    root = os.path.join(args.train_dataset, "train")
    train_dataset = datasets.ImageFolder(root, transform=train_transforms)
    root = os.path.join(args.train_dataset, "val")
    test_dataset = datasets.ImageFolder(root, transform=val_transforms)
    sub_idx = list(range(57600))
    train_dataset = Subset(train_dataset, sub_idx)
    train_sampler, test_sampler = None, None
    len_tdset = len(train_dataset)
    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False,
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        sampler=test_sampler,
        drop_last=False,
        num_workers=args.num_workers,
    )

    return train_loader, test_loader, len_tdset


def build_torchvision_loader_gpu(args):
    from dataloader.datasets import build_transform

    args.input_size = 224
    args.imagenet_default_mean_and_std = True
    args.train_interpolation = "bicubic"
    args.reprob = 0.20
    args.remode = "pixel"
    args.recount = 1
    args.aa = "rand-m9-mstd0.5-inc1"
    # args.aa = "rand-m9-mstd0.5"
    args.color_jitter = 0.4
    args.crop_pct = None  # 224 / 256

    train_transforms = build_transform(is_train=True, args=args)
    val_transforms = build_transform(is_train=False, args=args)

    '''
    print("---" * 10)
    print("Train Transforms:")
    for t in train_transforms.transforms:
        print(t)
    print("---" * 10)
    print("---" * 10)
    print("Val Transforms:")
    for t in val_transforms.transforms:
        print(t)
    print("---" * 10)
    '''

    root = os.path.join(args.train_dataset, "train")
    train_dataset = datasets.ImageFolder(root, transform=train_transforms)
    root = os.path.join(args.train_dataset, "val")
    test_dataset = datasets.ImageFolder(root, transform=val_transforms)
    sub_idx = list(range(57600))
    train_dataset = Subset(train_dataset, sub_idx)
    train_sampler, test_sampler = None, None
    len_tdset = len(train_dataset)
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True,
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=False,
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        sampler=test_sampler,
        drop_last=False,
        num_workers=args.num_workers,
    )

    return train_loader, test_loader, len_tdset
