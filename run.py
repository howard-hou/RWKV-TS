import os
import json
import argparse
from pathlib import Path
# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

from rwkv.model import RWKV
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

from dataloader import Dataset_ETT_hour
from pipeline import Pipeline
from experiment import ExpRWKV
from utils import set_random_seed
from visualize import visualize_experiment


def parse_arg():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('data_path', type=str, help='data path')
    arg_parser.add_argument('model_path', type=str, help='RWKV model path')
    arg_parser.add_argument('output_dir', type=str, help='output dir')
    arg_parser.add_argument('--strategy', type=str, default='cpu fp32')
    arg_parser.add_argument('--input_len', type=int, default=96)
    arg_parser.add_argument('--pred_len', type=int, default=24)
    arg_parser.add_argument('--seed', type=int, default=22)
    arg_parser.add_argument('--num_shots', type=int, default=1)
    arg_parser.add_argument('--features', type=str, default='S')
    arg_parser.add_argument('--target', type=str, default='OT')
    arg_parser.add_argument('--disable_scale', action='store_true')
    arg_parser.add_argument('--visualize', action='store_true')
    return arg_parser.parse_args()


def main():
    args = parse_arg()
    print(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    model = RWKV(model=args.model_path, strategy=args.strategy)
    tokenizer = TRIE_TOKENIZER('rwkv_vocab_v20230424.txt')
    pipeline = Pipeline(model, tokenizer)
    test_dataset = Dataset_ETT_hour(data_path=args.data_path, input_len=args.input_len, 
                               pred_len=args.pred_len, flag='test', features=args.features, 
                               target=args.target, scale=not args.disable_scale)
    train_dataset = Dataset_ETT_hour(data_path=args.data_path, input_len=args.input_len, 
                               pred_len=args.pred_len, flag='train', features=args.features, 
                               target=args.target, scale=not args.disable_scale)

    exp = ExpRWKV(pipeline, test_dataset, train_dataset, args.input_len, args.pred_len)
    # run predictions for visualization
    if args.visualize:
        visualize_experiment(exp, output_dir, col=0, k=args.num_shots, num_plots=6)
    else:
        # # run experiment
        exp_res = exp.run_univariate_test_exp(col=0, k=args.num_shots, 
                                              scale_transform=args.disable_scale)
        exp_out = {"exp_config": vars(args), "exp_res": exp_res}
        output_path = output_dir / 'exp_res.json'
        json.dump(exp_out, open(output_path, "w"), indent=4)


if __name__ == "__main__":
    main()