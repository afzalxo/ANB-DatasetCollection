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


def load_satellite_data(path):
    train_file = os.path.join(path, "satellite_train.npy")
    test_file = os.path.join(path, "satellite_test.npy")

    all_train_data, all_train_labels = (
        np.load(train_file, allow_pickle=True)[()]["data"],
        np.load(train_file, allow_pickle=True)[()]["label"],
    )
    test_data, test_labels = (
        np.load(test_file, allow_pickle=True)[()]["data"],
        np.load(test_file, allow_pickle=True)[()]["label"],
    )

    # rerange labels to 0-23
    all_train_labels = all_train_labels - 1
    test_labels = test_labels - 1

    # normalize data
    all_train_data = (
        all_train_data - all_train_data.mean(axis=1, keepdims=True)
    ) / all_train_data.std(axis=1, keepdims=True)
    test_data = (test_data - test_data.mean(axis=1, keepdims=True)) / test_data.std(
        axis=1, keepdims=True
    )

    # add dimension
    all_train_data = np.expand_dims(all_train_data, 1)
    test_data = np.expand_dims(test_data, 1)

    # convert to tensor/longtensor
    all_train_tensors, all_train_labeltensor = torch.from_numpy(all_train_data).type(
        torch.FloatTensor
    ), torch.from_numpy(all_train_labels).type(torch.LongTensor)

    test_tensors, test_labeltensor = torch.from_numpy(test_data).type(
        torch.FloatTensor
    ), torch.from_numpy(test_labels).type(torch.LongTensor)
    testset = data_utils.TensorDataset(test_tensors, test_labeltensor)

    trainset = data_utils.TensorDataset(all_train_tensors, all_train_labeltensor)

    return trainset, None, testset


class SatelliteFFCVDataLoader:
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
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(self.gpu)]
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
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(self.gpu)]
        loader = Loader(
            val_path,
            batch_size=batch_size,
            num_workers=self.num_workers,
            order=OrderOption.RANDOM,
            drop_last=False,
            pipelines={"covariate": image_pipeline, "label": label_pipeline},
        )
        return loader
