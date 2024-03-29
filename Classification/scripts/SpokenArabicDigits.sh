python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name SpokenArabicDigits \
    --records_file Classification_records.xls \
    --data_dir ./datasets/SpokenArabicDigits \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 50 \
    --lr 0.001 \
    --patch_size 16 \
    --stride 8 \
    --optimizer RAdam \
    --d_model 128 \
    --dropout 0.2 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy \
    --seed 22