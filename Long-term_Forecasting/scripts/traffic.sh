seq_len=512
model=RWKV4TS

for percent in 100
do
for pred_len in 96 192 336 720
do

python main.py \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 2048 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 128 \
    --n_heads 2 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 2 \
    --itr 3 \
    --model $model \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --pretrain 0

done
done
