########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from typing import Dict, List, Sequence, Any
import pandas as pd


class TestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.seq_len = args.ctx_len
        self.build_test_set()
        self.do_normalize = args.do_normalize

    def build_test_set(self):
        df = pd.read_excel(self.args.data_file, sheet_name=None)
        keys_sorted = sorted(k for k in df)
        data_df = df[keys_sorted[-1]]
        X = data_df["nwp_ws100"].to_numpy()[:, np.newaxis]
        y = data_df["fj_windSpeed"].to_numpy()[:, np.newaxis]
        # split X, y to chunk 
        X_chunks = [X[i:i+self.seq_len] for i in range(0, len(X), self.seq_len)]
        y_chunks = [y[i:i+self.seq_len] for i in range(0, len(y), self.seq_len)]
        print(f"input chunks: {len(X_chunks)}, target chunks: {len(y_chunks)}")
        self.X = X_chunks
        self.y = y_chunks

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.do_normalize:
            input_points = (self.X[idx] - self.args.X_mean) / self.args.X_std
        else:
            input_points = self.X[idx]
        targets = self.y[idx]
        return dict(input_points=input_points, targets=targets)


class TrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.seq_len = args.ctx_len
        self.label_smoothing = args.label_smoothing
        self.build_train_set()
        self.do_normalize = args.do_normalize

    def build_train_set(self):
        df = pd.read_excel(self.args.data_file, sheet_name=None)
        keys_sorted = sorted(k for k in df)
        data_df_list = [df[k] for k in keys_sorted[:-1]]
        if self.label_smoothing > 0:
            data_df_list_smooth = []
            for df in data_df_list:
                df["fj_windSpeed"] = df["fj_windSpeed"].rolling(window=self.label_smoothing,center=True).median()
                df.loc[df["fj_windSpeed"].isna(), "fj_windSpeed"] = 0
                df["fj_windSpeed"] = df["fj_windSpeed"].fillna(0)
                data_df_list_smooth.append(df)
            data_df = pd.concat(data_df_list_smooth)
            print(f"label smoothing with window={self.label_smoothing} applied")
        else:
            data_df = pd.concat(data_df_list)
            print(f"raw data used, no label smoothing applied")
        self.X = data_df["nwp_ws100"].to_numpy()[:, np.newaxis]
        self.y = data_df["fj_windSpeed"].to_numpy()[:, np.newaxis]
        print(f"input shape: {self.X.shape}, target shape: {self.y.shape}")
        self.X_mean = self.X.mean()
        self.X_std = self.X.std()
        self.y_mean = self.y.mean()
        self.y_std = self.y.std()

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        # random sample a start index
        s = random.randrange(len(self.X) - self.seq_len)
        if self.do_normalize:
            input_points = (self.X[s:s+self.seq_len] - self.X_mean) / self.X_std
            targets = (self.y[s:s+self.seq_len] - self.y_mean) / self.y_std
        else:
            input_points = self.X[s:s+self.seq_len]
            targets = self.y[s:s+self.seq_len]
        return dict(input_points=input_points, targets=targets)