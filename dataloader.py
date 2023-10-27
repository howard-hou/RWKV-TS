from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils import StandardScaler

import warnings
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, data_path, seq_len, pred_len, flag='train', 
                 features='S', target='OT', scale=True):
        # info
        self.seq_len = seq_len
        self.pred_len = pred_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)
        # [0, 1year - seq_len, 1year + 4month - seq_len]
        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        # [1year, 1year + 4month, 1year + 8month]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
if __name__ == "__main__":
    ETT_hour_dataset = Dataset_ETT_hour(data_path='data/ETT/ETTh2.csv', seq_len=96, 
                                        pred_len=24, flag='test', features='S', 
                                        target='OT', scale=True)
    from serialize import vec2str
    for i in range(1):
        print(ETT_hour_dataset[i][0].shape, ETT_hour_dataset[i][1].shape)
        pred = ETT_hour_dataset[i][0]
        pred_str = vec2str(pred[:,0])
        print(pred_str)
        print(len(pred_str.split(",")))
