import os
import torch
from torchvision import datasets, transforms


def build_torchvision_loader(args):
    is_train = False
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
    # sampler_val = torch.utils.data.DistributedSampler(dataset,
    #                num_replicas=args.world_size, rank=args.global_rank,
    #                shuffle=False)
    valid_queue = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.val_batch_size),
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return valid_queue, dataset
