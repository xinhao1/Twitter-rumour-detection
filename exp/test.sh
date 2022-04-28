#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/test.py \
        --output_dir nlppj \
        --pre_encoder cardiffnlp/twitter-roberta-base-2021-124m \
        --ckpt nlppj/f1_score=0.8547.ckpt \
        --eval_batch_size 32 \
        --test_batch_size 32 \
        --max_length 512 \
        --gpus 1 \
        --num_sanity_val_steps 0 \
        --num_workers 24 \
        --fp16