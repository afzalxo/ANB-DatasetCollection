import os
import numpy as np
import pickle

import torch
import torch as ch
import torch.utils.data as data_utils
from torch.utils.data import Dataset

from collections import Counter
from sklearn.model_selection import train_test_split

from typing import List


class ECGDataset(Dataset):
    def __init__(self, data, label, pid=None):
        self.data = data
        self.label = label
        self.pid = pid

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)


def read_data_physionet_4(path, window_size=1000, stride=500):
    # read pkl
    with open(os.path.join(path, 'challenge2017.pkl'), 'rb') as fin:
        res = pickle.load(fin)
    # scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std
    # encode label
    all_label = []
    for i in res['label']:
        if i == 'N':
            all_label.append(0)
        elif i == 'A':
            all_label.append(1)
        elif i == 'O':
            all_label.append(2)
        elif i == '~':
            all_label.append(3)
    all_label = np.array(all_label)

    # split train test
    X_train, X_test, Y_train, Y_test = train_test_split(
        all_data, all_label, test_size=0.1, random_state=0)

    # slide and cut
    print('before: ')
    print(Counter(Y_train), Counter(Y_test))
    X_train, Y_train = slide_and_cut(
        X_train, Y_train, window_size=window_size, stride=stride)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride,
                                             output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_test))

    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    #X_train = np.expand_dims(X_train, 1)
    #X_test = np.expand_dims(X_test, 1)

    trainset = ECGDataset(X_train, Y_train)
    testset = ECGDataset(X_test, Y_test, pid_test)

    return trainset, None, testset


def slide_and_cut(X, Y, window_size, stride, output_pid=False, datatype=4):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            if datatype == 4:
                i_stride = stride//6
            elif datatype == 2:
                i_stride = stride//10
            elif datatype == 2.1:
                i_stride = stride//7
        elif tmp_Y == 2:
            i_stride = stride//2
        elif tmp_Y == 3:
            i_stride = stride//20
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)


def ecg_dataloader_tv(path, batch_size):
    train_dataset, _, test_dataset = read_data_physionet_4('/home/aahmadaa/datasets/ecg/')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader, test_dataset
