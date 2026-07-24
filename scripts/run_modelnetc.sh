#!/bin/bash
# ModelNet40-C Robustness Evaluation
CUDA_VISIBLE_DEVICES=0 python run_modelnetc.py \
    --bz 32 \
    --points "[1024,512,256,128]" \
    --stages 4 \
    --k "[110,110,110,110]" \
    --metric 2 \
    --rescale 0.8 \
    --surface geobp \
    --data_root data/modelnet_c \
    --eval_corruptions scale jitter dropout_global rotate
