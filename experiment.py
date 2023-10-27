import torch.nn as nn
import numpy as np
from tqdm import tqdm
from metrics import calc_metrics
from serialize import get_univariate_kshot_examples, get_input_prompt, vec2str, output_str2list

class ExpRWKV():
    def __init__(self, pipeline, test_dataset, train_dataset, input_len, pred_len) -> None:
        self.pipeline = pipeline
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
        self.input_len = input_len
        self.pred_len = pred_len
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def run_test_exp(self, col=0, k=1):
        kshot_examples = get_univariate_kshot_examples(self.train_dataset, col=col, k=k)
        print("examples:\n"+kshot_examples)
        y_preds = []
        y_trues = []
        for i in tqdm(range(len(self.test_dataset)), desc='testing'):
            seq_x, y_true = self.test_dataset[i]
            input_prompt = get_input_prompt(seq_x, kshot_examples, col=col)
            num_token_to_generate = self.calc_generation_length(y_true, col=col)
            # print(input_prompt)
            output_str = self.pipeline.greedy_generate(input_prompt, 
                                                  token_count=num_token_to_generate)
            # print(output_str)
            # exit()
            y_pred = output_str2list(output_str, max_len=self.pred_len)
            y_preds.append(y_pred)
            y_trues.append(y_true[:, col])
        y_preds = np.array(y_preds)
        y_trues = np.array(y_trues)
        print('test shape:', y_preds.shape, y_trues.shape)
        mae, mse, rmse, mape, mspe = calc_metrics(y_preds, y_trues)
        print(mae, mse, rmse, mape, mspe)

    def calc_generation_length(self, seq_y, col=0):
        str_y = vec2str(seq_y[:, col])
        token_y = self.pipeline.encode(str_y)
        return len(token_y)