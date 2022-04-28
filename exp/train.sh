#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/train.py \
        --output_dir nlppj \
        --pre_encoder cardiffnlp/twitter-roberta-base-2021-124m \
        --warmup_ratio 0.1 \
        --learning_rate 5e-5 \
        --weight_decay 1e-3 \
        --gradient_clip_val 1 \
        --train_batch_size 16 \
        --eval_batch_size 16 \
        --test_batch_size 16 \
        --max_length 512 \
        --max_epochs 100 \
        --save_top_k 1 \
        --gpus 1 \
        --num_sanity_val_steps -1 \
        --log_every_n_steps 20 \
        --check_val_every_n_epoch 5 \
        --early_stopping_patience 4 \
        --num_workers 24 \
        --wandb \
        --fp16