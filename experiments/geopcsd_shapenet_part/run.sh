#!/usr/bin/env bash
set -euo pipefail
: "${TFCW_SHAPENET_PART_ROOT:?Set the ShapeNet-Part dataset directory}"
cd "$(dirname "$0")"
python -u run_part.py --bz 128 --points '[1024,512,256,128]' --stages 4 --k '[105,105,110,120]' --de_k 3 --metric 2 --rescale 0.2 --gamma 230 --surface geobp
