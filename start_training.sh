#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python3 train.py \
    --seed 63 \
    --dataset_name openvivqa \
    --epochs 40 \
    --patience 3 \
    --n_text_paras 2 \
    --text_para_thresh 0.8 \
    --n_text_para_pool 10 \
    --is_filter false \
    --is_text_augment false \
    --use_dynamic_thresh false \
    --start_threshold 0.8 \
    --min_threshold 0.0 \
    --is_log_result true