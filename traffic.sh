
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path traffic.csv \
#   --model_id traffic_$seq_len'_'96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --enc_in 862 \
#   --des 'Exp' \
#   --itr 1 --batch_size 16 --learning_rate 0.05 

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path traffic.csv \
#   --model_id traffic_$seq_len'_'192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 192 \
#   --enc_in 862 \
#   --des 'Exp' \
#   --itr 1 --batch_size 16 --learning_rate 0.05 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.05 \
  --apprx 1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.05 \
  --apprx 1
