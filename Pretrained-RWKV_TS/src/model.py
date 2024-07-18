########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
wkv6_cuda = load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
    
class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################

class RWKV_Tmix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            if args.n_embd >= 4096:
                D_MIX_LORA = 64
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            if args.n_embd >= 4096:
                D_DECAY_LORA = 128
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)

########################################################################################################

class RWKV_CMix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
    
########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        
    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optim_groups = [{"params": trainable_params, "weight_decay": self.args.weight_decay}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, x):
        args = self.args
        # B, T, D = x.size()
        # assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        if args.dropout > 0:
            x = self.drop0(x)

        for block in self.blocks:
            if args.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)

        x = self.ln_out(x)

        #x = self.head(x)

        return x

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

class Projector(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.conv3 = nn.Conv1d(1, n_embd//4, kernel_size=3, stride=1, groups=1, padding='same')
        self.conv5 = nn.Conv1d(1, n_embd//4, kernel_size=5, stride=1, groups=1, padding='same')
        self.conv7 = nn.Conv1d(1, n_embd//4, kernel_size=7, stride=1, groups=1, padding='same')
        self.conv9 = nn.Conv1d(1, n_embd//4, kernel_size=9, stride=1, groups=1, padding='same')
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.size()
        return self.drop(torch.cat([self.conv3(x), self.conv5(x), self.conv7(x), self.conv9(x)], 1)).view(B, T, -1)


class Projector2(nn.Module):
    def __init__(self, n_embd, shift_steps):
        super().__init__()
        self.conv3 = nn.Conv1d(1, n_embd//8, kernel_size=3, stride=1, groups=1, padding='same')
        self.conv5 = nn.Conv1d(1, n_embd//8, kernel_size=5, stride=1, groups=1, padding='same')
        self.conv7 = nn.Conv1d(1, n_embd//8, kernel_size=7, stride=1, groups=1, padding='same')
        self.conv9 = nn.Conv1d(1, n_embd//8, kernel_size=9, stride=1, groups=1, padding='same')
        self.linear = nn.Linear(shift_steps, n_embd//2)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, shifted_y):
        B, T, C = x.size()
        x = x.view(B, C, T)
        shifted_y_emb = self.linear(shifted_y)
        x_emb = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x), self.conv9(x)], 1)
        return self.drop(torch.cat([x_emb.view(B, T, -1), shifted_y_emb], 2))



class TimeSeriesRWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rwkv = RWKV(args)
        if args.load_model:
            self.load_rwkv_from_pretrained(args.load_model)
        self.proj = Projector2(args.n_embd, args.shift_steps)
        self.head = nn.Linear(args.n_embd, 1, bias=False)
        self.best_val_loss = torch.tensor(float("inf"))
        self.do_normalize = args.do_normalize
        self.prefix_len = args.prefix_len

    def load_rwkv_from_pretrained(self, path):
        self.rwkv.load_state_dict(torch.load(path, map_location="cpu"))
        rank_zero_info(f"Loaded pretrained RWKV from {path}")

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    
    def freeze_rwkv(self):
        # freeze all layers excluding the layernorms, projection and head
        self.rwkv.emb.requires_grad_(False)
        self.rwkv.head.requires_grad_(False)
        for block in self.rwkv.blocks:
            for name, param in block.named_parameters():
                if "ln" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        name_of_trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        rank_zero_info(f"Name of trainable parameters in optimizers: {name_of_trainable_params}")
        rank_zero_info(f"Number of trainable parameters in optimizers: {len(trainable_params)}")
        optim_groups = [{"params": trainable_params, "weight_decay": self.args.weight_decay}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    def forward(self, samples):
        x, y, shifted_y = samples["input_points"], samples["targets"], samples["shifted_targets"]
        x = self.proj(x, shifted_y)
        x = self.rwkv(x)[:, self.prefix_len:, :] # 
        outputs = self.head(x)
        return outputs, y

    def bidirectional_forward(self, x, x_emb=None):
        pass
    
    def training_step(self, batch, batch_idx):
        outputs, targets = self(batch)
        loss = F.mse_loss(outputs.view(-1), targets.view(-1))
        return loss
    
    def training_epoch_end(self, outputs):
        all_losses = [x['loss'] for x in outputs if 'loss' in x]
        train_loss = sum(all_losses) / len(all_losses)
        self.log("train_loss", train_loss, sync_dist=True)
        my_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("my_lr", my_lr, sync_dist=True)
    
    def validation_step(self, batch, batch_idx):
        outputs, targets = self(batch)
        if self.do_normalize:
            outputs = self.args.y_mean + self.args.y_std * outputs
        return dict(predicts=outputs.view(-1), targets=targets.view(-1))
    
    def validation_epoch_end(self, outputs):
        all_predicts = torch.cat([out["predicts"] for out in outputs])
        all_targets = torch.cat([out["targets"] for out in outputs])
        val_loss = F.mse_loss(all_predicts.view(-1), all_targets.view(-1)).item()
        self.log("val_loss", val_loss, sync_dist=True)
        if val_loss < self.best_val_loss and self.trainer.current_epoch >=1:
            # remove last best model
            if os.path.exists(os.path.join(self.args.proj_dir, f"best-{self.best_val_loss:.3f}.pth")):
                os.remove(os.path.join(self.args.proj_dir, f"best-{self.best_val_loss:.3f}.pth"))
            state_dict = self.state_dict()
            torch.save(state_dict, os.path.join(self.args.proj_dir, f"best-{val_loss:.3f}.pth"))
            self.best_val_loss = val_loss
            # plot best figure
            from .utils import plot_prediction_and_target
            plot_dir = os.path.join(self.args.proj_dir, "best_model_plot")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plot_prediction_and_target(outputs, plot_dir)
    
    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

    @torch.no_grad()
    def predict(self, input_points) -> list[int]:
        ''' 
        input_points: Tensor of shape (B, T, 1)
        outputs: Tensor of shape (B, T, 1)
        '''
        outputs = self.head(self.rwkv(self.proj(input_points)))
        return outputs
