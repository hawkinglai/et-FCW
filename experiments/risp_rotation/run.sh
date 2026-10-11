#!/usr/bin/env bash
set -euo pipefail
: "${TFCW_MODELNET_NORMAL_ROOT:?Set modelnet40_normal_resampled directory}"
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python -u run_cls_ri.py --dataset mn40 --split 1 --bz 32 --points '[1024,512,256,128]' --stages 4 --k '[95,80,100,95]' --metric 2 --rescale 0.8 --surface knnxyz
