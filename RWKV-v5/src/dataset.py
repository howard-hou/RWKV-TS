########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(args.data_file).select_dtypes(include=['float64', 'float32']).values
        self.data_size = len(self.data)
        self.input_len = args.input_len
        self.pred_len = args.pred_len
        rank_zero_info(f"Data has {self.data_size} values, input_len={self.input_len}, pred_len={self.pred_len}")
        self.apply_standardization()

    def apply_standardization(self):
        '''Standardize features by removing the mean and scaling to unit variance'''
        self.std = self.data.std(0)
        self.mean = self.data.mean(0)
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        idx = idx % self.data_size

        if idx + self.input_len + self.pred_len >= self.data_size:
            idx = random.randint(0, self.data_size - self.input_len - self.pred_len - 1)

        x = self.data[idx:idx+self.input_len]
        y = self.data[idx+self.input_len:idx+self.input_len+self.pred_len]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y
