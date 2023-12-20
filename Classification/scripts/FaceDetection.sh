python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name FaceDetection \
    --records_file Classification_records.xls \
    --data_dir ./datasets/FaceDetection \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 50 \
    --lr 0.001 \
    --patch_size 16 \
    --stride 16 \
    --optimizer RAdam \
    --d_model 128 \
    --dropout 0.1 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy \
    --seed 22