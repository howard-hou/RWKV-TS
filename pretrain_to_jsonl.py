'''
convert to jsonl format for RWKV 
https://github.com/Abel2076/json2binidx_tool/tree/main
'''
import json
import argparse
from pathlib import Path
from dataloader import Dataset_ETT_hour
from serialize import vec2str


def process_univariate(dataset, to_int=False):
    data_str = vec2str(dataset.data, sep=',', to_int=to_int)
    return data_str


def write_jsonl(data_str, dataset_name, output_path):
    data = {
        'dataset_name': dataset_name,
        'data': data_str
    }
    with open(output_path, 'w') as f:
        json.dump(data, f)


def main():
    args = parse_arg()
    dataset = Dataset_ETT_hour(data_path=args.data_path, input_len=args.input_len, 
                               pred_len=args.pred_len, flag='train', features=args.features, 
                               target=args.target, scale=not args.disable_scale)
    data_str = process_univariate(dataset)
    dataset_name = Path(args.data_path).stem
    write_jsonl(data_str, dataset_name=dataset_name, output_path=args.output_path)
    

def parse_arg():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('data_path', type=str, help='data path')
    arg_parser.add_argument('output_path', type=str, help='output dir')
    arg_parser.add_argument('--input_len', type=int, default=96)
    arg_parser.add_argument('--pred_len', type=int, default=24)
    arg_parser.add_argument('--features', type=str, default='S')
    arg_parser.add_argument('--target', type=str, default='OT')
    arg_parser.add_argument('--disable_scale', action='store_true')
    return arg_parser.parse_args()



if __name__ == "__main__":
    main()