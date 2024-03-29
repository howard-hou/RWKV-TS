
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model RWKV4TS \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --gpt_layer 2 \
  --d_model 128 \
  --n_heads 2 \
  --d_ff 128 \
  --patch_size 1 \
  --stride 1 \
  --enc_in 25 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10 