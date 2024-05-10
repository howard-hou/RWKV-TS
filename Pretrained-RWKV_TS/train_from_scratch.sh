export CUDA_VISIBLE_DEVICES=0

python train.py --load_model "" \
    --wandb "" --proj_dir out/dummy \
    --data_file /root/autodl-tmp/Pretrained-RWKV_TS/combined_data.xlsx \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 96 --epoch_steps 1000 --epoch_count 10 --epoch_begin 0 --epoch_save 10 \
    --micro_bsz 16 --accumulate_grad_batches 1 --n_layer 2 --n_embd 64 --pre_ffn 0 \
    --lr_init 2e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 0 \
    --freeze_rwkv 0 --exp_name 'scratch_emb64_2layer'\
    --enable_progress_bar True
