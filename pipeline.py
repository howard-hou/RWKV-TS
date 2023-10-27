import torch
import numpy as np
from torch.nn import functional as F

class Pipeline():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, x):
        return self.tokenizer.encode(x)
    
    def decode(self, x):
        return self.tokenizer.decode(x)
    
    def greedy_generate(self, ctx, token_count=100):
        all_tokens = []
        out_last = 0
        out_str = ''
        state = None
        for i in range(token_count):
            # forward
            tokens = self.encode(ctx) if i == 0 else [token]
            out, state = self.model.forward(tokens, state)
            # greedy
            token = int(out.argmax())
            all_tokens += [token]
            # output
            tmp = self.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                out_str += tmp
                out_last = i + 1
        return out_str