export CUDA_VISIBLE_DEVICES=0

for accumulate_grad_batches in 2 4 8
do
    # convert accumulate_grad_batches to to int
    bs=$((accumulate_grad_batches * 64))
    python train.py --load_model /root/autodl-tmp/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth \
        --wandb "" --proj_dir out/pretrain_unidirection_bs${bs}_ls24_prefix4 \
        --data_file /root/autodl-tmp/Pretrained-RWKV_TS/combined_data.xlsx \
        --data_type "json" --vocab_size 65536 \
        --ctx_len 96 --epoch_steps 100 --epoch_count 100 --epoch_begin 0 --epoch_save 0 \
        --micro_bsz 64 --accumulate_grad_batches $accumulate_grad_batches --n_layer 24 --n_embd 2048 --pre_ffn 0 \
        --lr_init 1e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
        --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
        --freeze_rwkv 1 --exp_name pretrain_unidirection_bs${bs}_ls24_prefix4 --label_smoothing 24 \
        --enable_progress_bar True
done