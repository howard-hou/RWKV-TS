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


    def load_data(self, data_dir):
        data_files = Path(data_dir).glob('*.csv')
        name2data = {}
        for data_file in data_files:
            name = data_file.stem
            data = pd.read_csv(data_file).select_dtypes(include=['float64', 'float32']).values
            # split data into train and test by dataset name
            if name.startswith('ETTh'): # 1 year
                data = data[:12 * 30 * 24]
            elif name.startswith('ETTm'): # 1 year
                data = data[:12 * 30 * 24 * 4]
            else: # 70% of data for training
                data = data[:int(len(data)*0.7)]
            name2data[name] = (data - data.mean(0)) / data.std(0)
        return name2data


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
        self.input_len = args.input_len
        self.pred_len = args.pred_len
        self.data = self.load_data(args.data_file)
        self.data_size = len(self.data)
        rank_zero_info(f"Dataset {Path(args.data_file).stem}, Data has {self.data_size} rows, input_len={self.input_len}, pred_len={self.pred_len}")


    def load_data(self, data_file):
        data = pd.read_csv(data_file).select_dtypes(include=['float64', 'float32']).values
        name = Path(self.args.data_file).stem
        # split data into train and test by dataset name
        if name.startswith('ETTh'): # 1 year
            train_data  = data[:12 * 30 * 24]
            test_data = data[12 * 30 * 24 + 4 * 30 * 24 - self.input_len: 12 * 30 * 24 + 8 * 30 * 24]
        elif name.startswith('ETTm'): # 1 year
            train_data  = data[:12 * 30 * 24 * 4]
            test_data = data[12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.input_len: 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else: # 70% of data for training
            train_data  = data[:int(len(data)*0.7)]
            test_data = data[-int(len(data)*0.2):]
        data = (test_data - train_data.mean(0)) / train_data.std(0)
        return data

    def __len__(self):
        return self.data_size - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_len]
        y = self.data[idx+self.input_len:idx+self.input_len+self.pred_len]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y