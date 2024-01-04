export CUDA_VISIBLE_DEVICES=0

seq_len=336
model=RWKV4TS

for percent in 100
do
for pred_len in 96
do

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 128 \
    --decay_fac 0.5 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 2 \
    --d_ff 768 \
    --dropout 1 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 3 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 20 \
    --pretrain 1 \
    --pretrain 0

done
done
