#!/usr/bin/env bash
set -euo pipefail
: "${TFCW_SHAPENET_DATALOADERS_ROOT:?Set directory containing data/shapenetcore_partanno_segmentation_benchmark_v0_normal}"
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python run_part.py --bz 128 --points '[1024,512,256,128]' --stages 4 --k '[105,105,110,120]' --de_k 3 --metric 2 --rescale 0.2 --gamma 230 --surface knnxyz
