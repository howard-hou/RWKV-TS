import os
import argparse
# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

from rwkv.model import RWKV
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

from dataloader import Dataset_ETT_hour
from pipeline import Pipeline
from experiment import ExpRWKV
from utils import set_random_seed


def parse_arg():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('data_path', type=str, default='data path')
    arg_parser.add_argument('model_path', type=str, default='RWKV model path')
    arg_parser.add_argument('--strategy', type=str, default='cpu fp32')
    arg_parser.add_argument('--seq_len', type=int, default=96)
    arg_parser.add_argument('--pred_len', type=int, default=24)
    arg_parser.add_argument('--seed', type=int, default=22)
    return arg_parser.parse_args()


def main():
    args = parse_arg()
    set_random_seed(args.seed)
    model = RWKV(model=args.model_path, strategy=args.strategy)
    tokenizer = TRIE_TOKENIZER('rwkv_vocab_v20230424.txt')
    pipeline = Pipeline(model, tokenizer)
    test_dataset = Dataset_ETT_hour(data_path=args.data_path, seq_len=args.seq_len, 
                               pred_len=args.pred_len, flag='test', features='S', 
                               target='OT', scale=True)
    train_dataset = Dataset_ETT_hour(data_path=args.data_path, seq_len=args.seq_len, 
                               pred_len=args.pred_len, flag='train', features='S', 
                               target='OT', scale=True)
    exp = ExpRWKV(pipeline, test_dataset, train_dataset, args.seq_len, args.pred_len)
    exp.run_test_exp(col=0, k=2)



if __name__ == "__main__":
    main()