python pretrain_to_jsonl.py data/ETT-small/ETTh1.csv jsonl_data/ETTh1.jsonl
python pretrain_to_jsonl.py data/ETT-small/ETTm1.csv jsonl_data/ETTm1.jsonl
python pretrain_to_jsonl.py data/electricity/electricity.csv jsonl_data/electricity.jsonl
python pretrain_to_jsonl.py data/exchange_rate/exchange_rate.csv jsonl_data/exchange_rate.jsonl
python pretrain_to_jsonl.py data/illness/national_illness.csv jsonl_data/illness.jsonl
python pretrain_to_jsonl.py data/traffic/traffic.csv jsonl_data/traffic.jsonl
python pretrain_to_jsonl.py data/weather/weather.csv jsonl_data/weather.jsonl

cat jsonl_data/ETTh1.jsonl \
    jsonl_data/ETTm1.jsonl \
    jsonl_data/electricity.jsonl \
    jsonl_data/exchange_rate.jsonl \
    jsonl_data/illness.jsonl \
    jsonl_data/traffic.jsonl \
    jsonl_data/weather.jsonl \
    > jsonl_data/pretrain.jsonl