'''
convert to jsonl format for RWKV 
https://github.com/Abel2076/json2binidx_tool/tree/main
'''
import json
import argparse
from pathlib import Path
from dataloader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from serialize import vec2str

name2dataset = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
}

def process_univariate(dataset, dataset_name, to_int=False):
    num_features = dataset.data.shape[1]
    output_list = []
    for i in range(num_features):
        data_list = dataset.data[:, i].tolist()
        data_str = vec2str(data_list, sep=',', to_int=to_int)
        output_list.append({
            'meta': f'{dataset_name}_feature{i}',
            'text': data_str
        })
    return output_list


def write_jsonl(output_list, output_path):
    with open(output_path, 'w') as f:
        for data in output_list:
            data = json.dumps(data, ensure_ascii=False)
            f.write(data + '\n')


def main():
    args = parse_arg()
    dataset_name = Path(args.data_path).stem
    Dataset = name2dataset[dataset_name] if dataset_name in name2dataset else Dataset_Custom
    dataset = Dataset(data_path=args.data_path, input_len=args.input_len, 
                      pred_len=args.pred_len, flag='train', features="M", 
                      scale=not args.disable_scale)
    dataset_name = Path(args.data_path).stem
    output_list = process_univariate(dataset, dataset_name)
    write_jsonl(output_list, output_path=args.output_path)
    

def parse_arg():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('data_path', type=str, help='data path')
    arg_parser.add_argument('output_path', type=str, help='output dir')
    arg_parser.add_argument('--input_len', type=int, default=96)
    arg_parser.add_argument('--pred_len', type=int, default=24)
    arg_parser.add_argument('--disable_scale', action='store_true')
    return arg_parser.parse_args()



if __name__ == "__main__":
    main()