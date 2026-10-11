#!/usr/bin/env bash
set -euo pipefail
: "${TFCW_MODELNET40_ROOT:?Set directory containing data/modelnet40_ply_hdf5_2048}"
: "${TFCW_MODELNETC_ROOT:?Set directory containing jitter_0.h5 through jitter_4.h5 and add_global_0.h5 through add_global_4.h5}"
cd "$(dirname "$0")"
for corruption in jitter add_global; do
  CUDA_VISIBLE_DEVICES=0 python -u run_cls_modelnetcv2.py --surface knnxyz --corruption "$corruption" --modes global local united --data_root "$TFCW_MODELNETC_ROOT"
done
