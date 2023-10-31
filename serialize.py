import numpy as np
import torch


def float2str(f, precision=2, base=0, to_int=False):
    f = round(f, precision) * (10 ** base)
    if to_int:
        return str(int(f))
    return str(f)

def vec2str(vec, precision=2, sep=',', base=0, to_int=False):
    return sep.join([float2str(v, precision, base=base, to_int=to_int) for v in vec])

def output_str2list(output_str, max_len, sep=',', base=0, to_int=False):
    # \n in output_str is replaced by sep
    if '\n' in output_str:
        output_str = output_str.replace('\n', sep)
    output_splits = output_str.split(sep)
    filtered_splits = []
    for s in output_splits:
        try:
            if to_int:
                s = int(s)
                s = s / (10 ** base)
            else:
                s = float(s)
            filtered_splits.append(s)
        except:
            pass
    truncated_splits = filtered_splits[:max_len]
    return [float(s) for s in truncated_splits]

def is_valid_output_str(output_str, max_len, sep=',', base=0, to_int=False):
    try:
        output_list = output_str2list(output_str, max_len, 
                                      sep=sep, base=base, to_int=to_int)
        if len(output_list) != max_len:
            return False
        return True
    except:
        return False

def get_univariate_kshot_examples(train_dataset, col=0, k=1, precision=2, base=0, to_int=False):
    examples = ""
    for _ in range(k):
        seq_x, seq_y = train_dataset[np.random.randint(len(train_dataset))]
        str_x = vec2str(seq_x[:,col], precision=precision, base=base, to_int=to_int)
        str_y = vec2str(seq_y[:,col], precision=precision, base=base, to_int=to_int)
        examples += "Input:\n" + str_x + "\n" + "Output:\n" + str_y + "\n"
    return examples

def get_input_prompt(seq_x, examples, col=0, precision=2, base=0, to_int=False):
    input_str = "Input:\n" + vec2str(seq_x[:,col], precision=precision, base=base, to_int=to_int) + "\n" + "Output:\n"
    return examples + input_str