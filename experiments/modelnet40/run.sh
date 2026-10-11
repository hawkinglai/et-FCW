#!/usr/bin/env bash
set -euo pipefail
: "${TFCW_MODELNET40_ROOT:?Set directory containing data/modelnet40_ply_hdf5_2048}"
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python run_cls.py --dataset mn40 --split 1 --bz 32 --points '[1024,512,256,128]' --stages 4 --k '[66,66,66,66]' --metric 2 --rescale 0.8 --surface knnxyz
