export CUDA_VISIBLE_DEVICES=0

input_len=336
pred_len=24

for steps in 1000 5000 10000 50000 100000
do
python train.py --load_model "/houhaowen/huggingface_models/BlinkDL/rwkv-5-world/RWKV-5-World-1B5-v2-20231025-ctx4096.pth" \
    --wandb "" --proj_dir "out/step${steps}_input${input_len}_pred${pred_len}" \
    --data_dir "../data/ETT-small/" \
    --ctx_len 336 --epoch_steps $steps --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 32 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 \
    --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy "deepspeed_stage_2" --grad_cp 0 \
    --input_len $input_len --pred_len $pred_len
done