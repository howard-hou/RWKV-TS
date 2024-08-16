import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.micro_step = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        # global_step is update step, influence by gradient accumulation
        real_step = trainer.global_step * args.accumulate_grad_batches + args.epoch_begin * args.epoch_steps

        # LR schedule, cosine with warmup
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_total = (args.epoch_begin + args.epoch_count) * args.epoch_steps
            progress = (real_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            # cosine decay
            cosine_decay = max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
            lr = args.lr_final + (args.lr_init - args.lr_final) * cosine_decay 

        if real_step < w_step:
            lr = lr * (0.1 + 0.9 * real_step / w_step)

        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            else:
                param_group["lr"] = lr

        self.micro_step += 1