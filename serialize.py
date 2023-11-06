import numpy as np

PRECISION = 2 # constant

def float2str(f, to_int=False):
    f = round(f, PRECISION)
    if to_int:
        f = f * (10 ** PRECISION)
        return str(int(f))
    return str(round(f, PRECISION))

def vec2str(vec, sep=',', to_int=False):
    return sep.join([float2str(v, to_int=to_int) for v in vec])

def output_str2list(output_str, max_len, sep=',', to_int=False):
    # \n in output_str is replaced by sep
    if '\n' in output_str:
        output_str = output_str.replace('\n', sep)
    output_splits = output_str.split(sep)
    filtered_splits = []
    for s in output_splits:
        try:
            if to_int:
                s = int(s)
                s = s / (10 ** PRECISION)
            else:
                s = float(s)
            filtered_splits.append(s)
        except:
            pass
    truncated_splits = filtered_splits[:max_len]
    return [float(s) for s in truncated_splits]

def is_valid_output_str(output_str, max_len, sep=',', to_int=False):
    try:
        output_list = output_str2list(output_str, max_len, 
                                      sep=sep, to_int=to_int)
        if len(output_list) != max_len:
            return False
        return True
    except:
        return False

def get_univariate_kshot_examples(train_dataset, col=0, k=1, to_int=False):
    examples = ""
    for _ in range(k):
        seq_x, seq_y = train_dataset[np.random.randint(len(train_dataset))]
        str_x = vec2str(seq_x[:,col], to_int=to_int)
        str_y = vec2str(seq_y[:,col], to_int=to_int)
        examples += "Input:\n" + str_x + "\n" + "Output:\n" + str_y + "\n"
    return examples

def get_input_prompt(seq_x, examples, col=0, to_int=False):
    input_str = "Input:\n" + vec2str(seq_x[:,col], to_int=to_int) + "\n" + "Output:\n"
    return examples + input_str