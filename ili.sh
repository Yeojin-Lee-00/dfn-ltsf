#!/bin/bash

# Create directories if they do not exist
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=104
model_name=DLinear
apprx_tgt=seasonal

seed=100

# for seed in {1..100}
# do

python3 -u run_longExp.py \
  --is_training 1 \
  --root_path ./all_six_datasets/ \
  --data_path national_illness.csv \
  --model_id national_illness_${seq_len}_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 24 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --apprx 1 \
  --apprx_target $apprx_tgt \
  --seed $seed \
  --soft_flag 1 \
  --n_func 1 \
  --beta_alter 0 \
  --alpha_mult 2

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path national_illness.csv \
#   --model_id national_illness_${seq_len}_36 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 18 \
#   --pred_len 36 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --batch_size 32 \
#   --learning_rate 0.01 \
#   --apprx 1 \
#   --apprx_target $apprx_tgt \
#   --seed $seed

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path national_illness.csv \
#   --model_id national_illness_${seq_len}_48 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 18 \
#   --pred_len 48 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --batch_size 32 \
#   --learning_rate 0.01 \
#   --apprx 1 \
#   --apprx_target $apprx_tgt \
#   --seed $seed
  
# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path national_illness.csv \
#   --model_id national_illness_${seq_len}_60 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 18 \
#   --pred_len 60 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --batch_size 32 \
#   --learning_rate 0.01 \
#   --apprx 1 \
#   --apprx_target $apprx_tgt \
#   --seed $seed
# # done