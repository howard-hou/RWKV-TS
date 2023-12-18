import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from metrics import calc_metrics
from serialize import get_univariate_kshot_examples, get_input_prompt, vec2str, output_str2list, is_valid_output_str


class ExpRWKV():
    def __init__(self, model, test_dataset, train_dataset, input_len, pred_len) -> None:
        self.model = model
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
        self.input_len = input_len
        self.pred_len = pred_len
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def run_multivariate_exp(self):
        y_preds = []
        y_trues = []
        num_total_samples = len(self.test_dataset)
        num_total_samples = 10
        for i in tqdm(range(num_total_samples), desc='testing'):
            seq_x, y_true = self.test_dataset[i]
            seq_x = torch.from_numpy(seq_x).unsqueeze(0)
            y_pred = self.model(seq_x)
            y_preds.append(y_pred)
            y_trues.append(y_true)
        y_preds = np.array(y_preds)
        y_trues = np.array(y_trues)
        num_valid_samples = len(y_preds)
        print("test values:", y_preds[0,0:5], y_trues[0,0:5])
        print('test shape:', y_preds.shape, y_trues.shape)
        mae, mse, rmse, mape, mspe = calc_metrics(y_preds, y_trues)
        print(mae, mse, rmse, mape, mspe)
        return {"num_valid_samples": num_valid_samples, 
                "num_total_samples":num_total_samples, 
                "mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}

    def run_one_univariate_predict(self, i, col=0, k=1, to_int=False):
        kshot_examples = get_univariate_kshot_examples(self.train_dataset, col=col, k=k,
                                                        to_int=to_int)
        print("examples:\n"+kshot_examples)
        seq_x, y_true = self.test_dataset[i]
        input_prompt = get_input_prompt(seq_x, kshot_examples, col=col, to_int=to_int)
        num_tokens_to_generate = self.calc_generation_tokens(y_true, col=col, 
                                                             redundant=self.pred_len)
        print("input_prompt:\n"+input_prompt)
        output_str = self.pipeline.greedy_generate(input_prompt, 
                                                   token_count=num_tokens_to_generate)
        print("output:\n"+output_str)
        if is_valid_output_str(output_str, max_len=self.pred_len, to_int=to_int):
                y_pred = output_str2list(output_str, max_len=self.pred_len, to_int=to_int)
        else:
            print(f"{i} - invalid output_str: {output_str}")
            return None
        print("y_pred:", y_pred[:5])
        return seq_x[:, 0], y_pred, y_true[:, col]