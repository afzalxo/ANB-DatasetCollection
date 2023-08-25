import os
import numpy as np

import torch
import torch as ch
import torch.utils.data as data_utils

from typing import List

from ffcv.fields.decoders import IntDecoder, NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    Convert,
    ToDevice,
    ToTensor,
)


def load_deepsea_data(path):
    data = np.load(os.path.join(path, "deepsea_filtered.npz"))
    all_train_data = data["x_train"]
    all_train_labels = data["y_train"]
    test_data = data["x_test"]
    test_labels = data["y_test"]

    # convert to tensor/longtensor
    all_train_tensors, all_train_labeltensor = torch.from_numpy(all_train_data).type(
        torch.FloatTensor
    ), torch.from_numpy(all_train_labels).type(torch.FloatTensor)

    test_tensors, test_labeltensor = torch.from_numpy(test_data).type(
        torch.FloatTensor
    ), torch.from_numpy(test_labels).type(torch.FloatTensor)

    testset = data_utils.TensorDataset(test_tensors, test_labeltensor)
    trainset = data_utils.TensorDataset(all_train_tensors, all_train_labeltensor)

    return trainset, testset


def deepsea_dataloader_tv(path, batch_size):
    train_dataset, test_dataset = load_deepsea_data(path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
    return train_loader, test_loader


class DeepSEAFFCVDataLoader:
    def __init__(self, gpu, num_workers):
        self.gpu = ch.device(f"cuda:{gpu}")
        self.num_workers = num_workers

    def create_train_loader(self, train_path, batch_size):
        image_pipeline: List[Operation] = [
            NDArrayDecoder(),
            ToTensor(),
            ToDevice(self.gpu),
            Convert(ch.float16),
        ]
        label_pipeline: List[Operation] = [
            NDArrayDecoder(),
            ToTensor(),
            ToDevice(self.gpu),
            Convert(ch.float16),
        ]
        loader = Loader(
            train_path,
            batch_size=batch_size,
            num_workers=self.num_workers,
            order=OrderOption.RANDOM,
            drop_last=True,
            pipelines={"covariate": image_pipeline, "label": label_pipeline},
        )
        return loader

    def create_val_loader(self, val_path, batch_size):
        image_pipeline: List[Operation] = [
            NDArrayDecoder(),
            ToTensor(),
            ToDevice(self.gpu),
            Convert(ch.float16),
        ]
        label_pipeline: List[Operation] = [
            NDArrayDecoder(),
            ToTensor(),
            ToDevice(self.gpu),
            Convert(ch.float16),
        ]
        loader = Loader(
            val_path,
            batch_size=batch_size,
            num_workers=self.num_workers,
            order=OrderOption.RANDOM,
            drop_last=False,
            pipelines={"covariate": image_pipeline, "label": label_pipeline},
        )
        return loader
