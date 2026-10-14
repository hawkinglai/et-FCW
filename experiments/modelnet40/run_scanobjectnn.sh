#!/usr/bin/env bash
# t-FCW xyz (knnxyz, per-sample std) on ScanObjectNN. Same runner and model as run.sh; only the dataset differs.
set -euo pipefail
: "${TFCW_SCANOBJECTNN_DATALOADERS_ROOT:?Set path to the dataloaders directory containing data/h5_files}"
case "${1:-}" in
  obj_bg) split=1; batch=64 ;;
  obj_only) split=2; batch=64 ;;
  pb_t50_rs) split=3; batch=32 ;;
  *) printf 'Usage: bash run_scanobjectnn.sh {obj_bg|obj_only|pb_t50_rs}\n' >&2; exit 2 ;;
esac
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=0 python -u run_cls.py --dataset scan --split "$split" --bz "$batch" --points '[1024,512,256,128]' --stages 4 --k '[90,90,90,90]' --metric 2 --rescale 0.8 --surface knnxyz
