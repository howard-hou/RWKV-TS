########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info



class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.data = self.load_data(args.data_dir)
        self.data_size = sum([len(v) for v in self.data.values()])
        self.dataset_weight = [len(v)/self.data_size for v in self.data.values()]
        self.dataset_names = [k for k in self.data]
        self.input_len = args.input_len
        self.pred_len = args.pred_len
        rank_zero_info(f"Data has {len(self.data)} datasets, {self.data_size} rows, input_len={self.input_len}, pred_len={self.pred_len}")
        self.apply_standardization()


    def load_data(self, data_dir):
        data_files = Path(data_dir).glob('*.csv')
        name2data = {}
        for data_file in data_files:
            name = data_file.stem
            data = pd.read_csv(data_file).select_dtypes(include=['float64', 'float32']).values
            # use 90% of data for training, 10% for testing
            name2data[name] = data[:int(len(data)*0.9)]
        return name2data


    def apply_standardization(self):
        '''Standardize features by removing the mean and scaling to unit variance'''
        for name, data in self.data.items():
            self.data[name] = (data - data.mean(0)) / data.std(0)

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        # pick a dataset by dataset_weight
        dataset_idx = np.random.choice(len(self.dataset_weight), p=self.dataset_weight)
        dataset_name = self.dataset_names[dataset_idx]
        dataset = self.data[dataset_name]
        data_size = len(dataset)
        idx = random.randint(0, data_size - self.input_len - self.pred_len - 1)

        x = dataset[idx:idx+self.input_len]
        y = dataset[idx+self.input_len:idx+self.input_len+self.pred_len]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y


class TestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.data = self.load_data(args.data_file)
        self.data_size = len(self.data)
        self.input_len = args.input_len
        self.pred_len = args.pred_len
        rank_zero_info(f"Dataset {Path(args.data_file).stem}, Data has {self.data_size} rows, input_len={self.input_len}, pred_len={self.pred_len}")


    def load_data(self, data_file):
        data = pd.read_csv(data_file).select_dtypes(include=['float64', 'float32']).values
        # use 90% of data for training, 10% for testing
        data =  (data - data.mean(0)) / data.std(0)
        data = data[int(len(data)*0.9):]
        return data

    def __len__(self):
        return self.data_size - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_len]
        y = self.data[idx+self.input_len:idx+self.input_len+self.pred_len]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y