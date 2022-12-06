from typing import List

from ffcv.transforms import Squeeze, NormalizeImage

# from ffcv.transforms.common import Squeeze
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.pipeline.operation import Operation

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from pathlib import Path
import torch
import numpy as np
import torchvision


class ImageNetDataLoaders:
    def __init__(self, gpu, num_workers, distributed, in_memory, args):
        self.num_workers = num_workers
        self.distributed = distributed
        self.in_memory = in_memory
        self.gpu = gpu
        self.decoder = None
        self.args = args

    def create_train_loader(self, train_path, batch_size):
        this_device = f"cuda:{self.gpu}"
        train_loc = Path(train_path)
        assert train_loc.is_file()

        res = (
            self.args.min_res
        )  # self.args.input_size #get_resolution(0, self.args.min_res, self.args.max_res, self.args.end_ramp, self.args.start_ramp)
        scale = tuple((0.08, 1.0))
        ratio = tuple((3.0 / 4.0, 4.0 / 3.0))
        # self.rr = RandomResizedCropAndInterpolation(res, scale=scale, ratio=ratio, interpolation='bicubic')
        self.decoder = RandomResizedCropRGBImageDecoder(
            output_size=(res, res), scale=scale, ratio=ratio
        )
        # re_prob, re_mode, re_count, re_num_splits = 0.25, "pixel", 1, 0
        # self.re = RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu')
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(flip_prob=0.5),
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(self.args.IMAGENET_MEAN, self.args.IMAGENET_STD, np.float16),
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device), non_blocking=True),
        ]

        order = OrderOption.RANDOM if self.distributed else OrderOption.QUASI_RANDOM
        trainloader = Loader(
            train_path,
            batch_size=batch_size,
            num_workers=self.num_workers,
            order=order,
            os_cache=self.in_memory,
            drop_last=True,
            pipelines={"image": image_pipeline, "label": label_pipeline},
            distributed=self.distributed,
            seed=0,
        )
        return trainloader

    def create_val_loader(self, val_path, batch_size, resolution):
        this_device = f"cuda:{self.gpu}"
        val_loc = Path(val_path)
        assert val_loc.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(
            res_tuple, ratio=self.args.DEFAULT_CROP_RATIO
        )
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(self.args.IMAGENET_MEAN, self.args.IMAGENET_STD, np.float16),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device), non_blocking=True),
        ]
        valloader = Loader(
            val_path,
            batch_size=int(batch_size),
            num_workers=self.num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={"image": image_pipeline, "label": label_pipeline},
            distributed=self.distributed,
            seed=0,
        )

        return valloader


def get_resolution(epoch, min_res, max_res, end_ramp, start_ramp):
    assert min_res <= max_res

    if epoch <= start_ramp:
        return min_res

    if epoch >= end_ramp:
        return max_res

    # otherwise, linearly interpolate to the nearest multiple of 32
    interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
    final_res = int(np.round(interp[0] / 32)) * 32
    return final_res


class CIFAR10DataLoaders:
    def __init__(self, gpu, num_workers, distributed, in_memory, args):
        self.num_workers = num_workers
        self.distributed = distributed
        self.in_memory = in_memory
        self.gpu = gpu
        self.decoder = None
        self.args = args

    def get_loaders(self, train_path, batch_size=512):
        this_device = f"cuda:{self.gpu}"
        CIFAR_MEAN = [125.307, 122.961, 113.8575]
        CIFAR_STD = [51.5865, 50.847, 51.255]

        BATCH_SIZE = batch_size

        loaders = {}
        for name in ["train", "test"]:
            label_pipeline: List[Operation] = [
                IntDecoder(),
                ToTensor(),
                ToDevice(torch.device(this_device), non_blocking=True),
                Squeeze(),
            ]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

            # Add image transforms and normalization
            if name == "train":
                image_pipeline.extend(
                    [
                        RandomHorizontalFlip(),
                        RandomTranslate(padding=2),
                        Cutout(
                            8, tuple(map(int, CIFAR_MEAN))
                        ),  # Note Cutout is done before normalization.
                    ]
                )
            image_pipeline.extend(
                [
                    ToTensor(),
                    ToDevice(torch.device(this_device), non_blocking=True),
                    # ToDevice('cuda:0', non_blocking=True),
                    ToTorchImage(),
                    Convert(torch.float16),
                    torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ]
            )

            # Create loaders
            bs = BATCH_SIZE if name == "train" else BATCH_SIZE * 12
            loaders[name] = Loader(
                train_path + f"/cifar_{name}.beton",
                batch_size=bs,
                num_workers=8,
                order=OrderOption.RANDOM,
                drop_last=(name == "train"),
                distributed=self.distributed,
                seed=0,
                pipelines={"image": image_pipeline, "label": label_pipeline},
            )
        return loaders


def get_ffcv_loaders(local_rank, args):
    dl = ImageNetDataLoaders(
        gpu=local_rank,
        num_workers=args.num_workers,
        distributed=args.distributed,
        in_memory=args.in_memory,
        args=args,
    )
    train_queue, valid_queue = None, None
    if hasattr(args, "train_dataset"):
        train_queue = dl.create_train_loader(
            train_path=args.train_dataset, batch_size=args.train_batch_size
        )
    if hasattr(args, "val_dataset"):
        valid_queue = dl.create_val_loader(
            args.val_dataset,
            batch_size=args.val_batch_size,
            resolution=args.val_resolution,
        )
    return train_queue, valid_queue, dl
