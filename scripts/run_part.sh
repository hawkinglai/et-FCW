#!/bin/bash
# ShapeNet Part Segmentation (knnxyz, metric=2, rescale=0.2)
# Expected: ~70.4% mIoU
CUDA_VISIBLE_DEVICES=0 python run_part.py \
    --bz 128 \
    --points "[1024,512,256,128]" \
    --stages 4 \
    --k "[105,105,110,120]" \
    --metric 2 \
    --rescale 0.2 \
    --surface knnxyz \
    --de_k 3 \
    --gamma 230
