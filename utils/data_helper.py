# -----------------------------------------------------------------------------
# Copyright (c) 2025, Guowei Zhang
# All rights reserved.
# 
# This source code is licensed under the MIT License found in the LICENSE file
# in the root directory of this source tree.
# -----------------------------------------------------------------------------

import os
import numpy as np
import os.path
import torch.utils.data
import utils
import torchvision.transforms as transforms


def default_image_loader(path):
    return utils.load_pickle(path)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data, data_index, transform=None, shuffle=False):
        self.x = x_data[data_index]
        self.y = y_data[data_index]
        self.transform = transform
        self.shuffle = shuffle

    def __getitem__(self, index):
        x_one = self.x[index]
        y_one = self.y[index]
        x_one = torch.tensor(x_one, dtype=torch.float32)
        if self.transform is not None:
            x_one = self.transform(x_one)
        return x_one, y_one

    def __len__(self):
        return len(self.y)

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.y)).tolist())
        else:
            return iter(range(len(self.y)))


class DatasetSPAT(torch.utils.data.Dataset):
    def __init__(self, x_data_spa, x_data_tempo, y_data, data_index, transform=None, shuffle=False):
        self.x_spa = x_data_spa[data_index]
        self.x_tempo = x_data_tempo[data_index]
        self.y = y_data[data_index]
        self.transform = transform
        self.shuffle = shuffle

    def __getitem__(self, index):
        x_one_spa = self.x_spa[index]
        x_one_tempo = self.x_tempo[index]
        y_one = self.y[index]
        x_one_spa = torch.tensor(x_one_spa, dtype=torch.float32)
        x_one_tempo = torch.tensor(x_one_tempo, dtype=torch.float32)
        if self.transform is not None:
            x_one_cnn = self.transform(x_one_cnn)
        return x_one_spa, x_one_tempo, y_one

    def __len__(self):
        return len(self.y)

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.y)).tolist())
        else:
            return iter(range(len(self.y)))
