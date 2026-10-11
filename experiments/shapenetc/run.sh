#!/usr/bin/env bash
set -euo pipefail
: "${TFCW_SHAPENET_DATALOADERS_ROOT:?Set directory containing data/shapenetcore_partanno_segmentation_benchmark_v0_normal}"
: "${TFCW_SHAPENETC_ROOT:?Set directory containing clean.h5 and jitter/add_global severity files}"
cd "$(dirname "$0")"
for corruption in jitter add_global; do
  CUDA_VISIBLE_DEVICES=0 python -u run_part_cv2.py --bz 128 --points '[1024,512,256,128]' --stages 4 --k '[105,105,110,120]' --metric 2 --rescale 0.2 --de_k 3 --gamma 230 --data_root "$TFCW_SHAPENETC_ROOT" --corruption "$corruption"
done
