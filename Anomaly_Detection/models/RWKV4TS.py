
"""
RWKV "x051a" model - does not require custom CUDA kernel to train :)

References:
https://github.com/BlinkDL/RWKV-LM

Inference:
Always fast, and VRAM will not grow, because RWKV does not need KV cache.

Training:
Because we are not using custom CUDA kernel here, training is slightly slower than gpt+flash_attn when ctxlen is short.
Training becomes faster than gpt+flash_attn when ctxlen is long.
"""


import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from einops import rearrange
from layers.Embed import DataEmbedding

import math, warnings
import inspect
from dataclasses import dataclass


@dataclass
class RWKVConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RWKV_TimeMix_x051a(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd

            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.unsqueeze(-1))

            tmp = torch.zeros(self.n_head)
            for h in range(self.n_head):
                tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
            self.time_faaaa = nn.Parameter(tmp.unsqueeze(-1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.gate = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.ln_x = nn.GroupNorm(self.n_head, config.n_embd, eps=(1e-5)*64)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        H, N = self.n_head, self.head_size
        #
        # we divide a block into chunks to speed up computation & save vram.
        # you can try to find the optimal chunk_len for your GPU.
        # avoid going below 128 if you are using bf16 (otherwise time_decay might be less accurate).
        #
        if T % 256 == 0: Q = 256
        elif T % 128 == 0: Q = 128
        else:
            Q = T
            warnings.warn(f'\n{"#"*80}\n\n{" "*38}Note\nThe GPT-mode forward() should only be called when we are training models.\nNow we are using it for inference for simplicity, which works, but will be very inefficient.\n\n{"#"*80}\n')
        assert T % Q == 0

        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xr = x + xx * self.time_maa_r
        xg = x + xx * self.time_maa_g
        r = self.receptance(xr).view(B, T, H, N).transpose(1, 2) # receptance
        k = self.key(xk).view(B, T, H, N).permute(0, 2, 3, 1) # key
        v = self.value(xv).view(B, T, H, N).transpose(1, 2) # value
        g = F.silu(self.gate(xg)) # extra gate

        w = torch.exp(-torch.exp(self.time_decay.float())) # time_decay
        u = self.time_faaaa.float() # time_first

        ws = w.pow(Q).view(1, H, 1, 1)

        ind = torch.arange(Q-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, Q).pow(ind)

        wk = w.view(1, H, 1, Q)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, Q))
        w = torch.tile(w, [Q])
        w = w[:, :-Q].view(-1, Q, 2*Q - 1)
        w = w[:, :, Q-1:].view(1, H, Q, Q)

        w = w.to(dtype=r.dtype) # the decay matrix
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)

        state = torch.zeros(B, H, N, N, device=r.device, dtype=r.dtype) # state
        y = torch.empty(B, H, T, N, device=r.device, dtype=r.dtype) # output

        for i in range(T // Q): # the rwkv-x051a operator
            rr = r[:, :, i*Q:i*Q+Q, :]
            kk = k[:, :, :, i*Q:i*Q+Q]
            vv = v[:, :, i*Q:i*Q+Q, :]
            y[:, :, i*Q:i*Q+Q, :] = ((rr @ kk) * w) @ vv + (rr @ state) * wb
            state = ws * state + (kk * wk) @ vv

        y = y.transpose(1, 2).contiguous().view(B * T, C)
        y = self.ln_x(y).view(B, T, C) * g

        # output projection
        y = self.dropout(self.output(y))
        return y

class RWKV_ChannelMix_x051a(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.value = nn.Linear(3 * config.n_embd, config.n_embd, bias=config.bias)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        x = self.key(xk)
        x = torch.relu(x) ** 2
        x = self.value(x)
        x = torch.sigmoid(self.receptance(xr)) * x
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.tmix = RWKV_TimeMix_x051a(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.cmix = RWKV_ChannelMix_x051a(config, layer_id)

    def forward(self, x):
        x = x + self.tmix(self.ln_1(x))
        x = x + self.cmix(self.ln_2(x))
        return x


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.d_ff = configs.d_ff
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(configs.enc_in * self.patch_size, configs.d_model, 
                                           configs.embed, configs.freq, configs.dropout)
        
        rwkv_config = RWKVConfig(n_layer=configs.gpt_layers, 
                                    n_head=configs.n_heads, 
                                    n_embd=configs.d_model)
        self.rwkv = nn.ModuleList([Block(rwkv_config, i) for i in range(rwkv_config.n_layer)])
        print("rwkv = {}".format(self.rwkv))
        
        self.ln_proj = nn.LayerNorm(configs.d_ff)
        self.out_layer = nn.Linear(
                configs.d_ff, 
                configs.c_out, 
                bias=True)

        device = "cuda:0" if configs.use_gpu else "cpu"
        for layer in (self.rwkv, self.ln_proj, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer
        seg_num = 25
        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')

        enc_out = torch.nn.functional.pad(x_enc, (0, self.configs.d_model - x_enc.shape[-1]))

        dec_out = enc_out
        for block in self.rwkv:
            dec_out = block(dec_out)

        outputs = dec_out[:, :, :self.d_ff]
        dec_out = self.out_layer(outputs)
        # De-Normalization from Non-stationary Transformer
        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)
        dec_out = dec_out * \
                  (stdev[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = dec_out + \
                  (means[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')
        return dec_out
