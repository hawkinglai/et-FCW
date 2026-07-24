#!/bin/bash
# ModelNet40 Classification (knnxyz, metric=1, rescale=0.2)
# Best: 84.64% with k=120 (paper: 84.8%, diff: -0.16%)
CUDA_VISIBLE_DEVICES=0 python run_cls.py \
    --dataset mn40 \
    --bz 32 \
    --points "[1024,512,256,128]" \
    --stages 4 \
    --k "[120,120,120,120]" \
    --metric 1 \
    --rescale 0.2 \
    --surface knnxyz
