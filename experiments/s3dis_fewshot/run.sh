#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_ID="${GPU_ID:-0}"

####################  Dataset and Model Parameters  #######################
MODEL='tfcw'
STAGES=4
KNBRS="[50,50,50,50]"
DE_K=20
RESCALE=0.2
METRIC=2 # euclidean seuclidean manhattan
SURFACE="${SURFACE:-geobp}" # knnxyz geobp
DATASET='s3dis' # s3dis scannet
CVFOLD="${CVFOLD:-0}"
DATA_PATH="${TFCW_S3DIS_ROOT:-dataloaders/data/S3DIS/scenes/blocks_bs1_s1/}"
SAVE_PATH="${SAVE_PATH:-./log_s3dis/}"

####################  Optimization Parameters  #######################
BATCH_SIZE=1
N_WORKERS=8

####################  Few-Shot Episode Setting  #######################
N_WAY="${N_WAY:-2}"
K_SHOT="${K_SHOT:-1}"
N_QUERIES=1
N_EPISODE_TEST=100

####################  Point Cloud Processing  #######################
POINTS="${POINTS:-[1024,512,256,128]}"
PC_ATTRIBS='xyzrgbXYZ'
WAY_PCRATIO='[0.15, 0.15]'
WAY_PCNUM='[300, 300]'

### prediction
GAMMA="${GAMMA:-1}"
# GAMMA=31
# knnxyz = 31

args=(--model "${MODEL}" --stages $STAGES --k_neighbors "${KNBRS}" --de_k $DE_K
      --rescale $RESCALE --metric "${METRIC}" --surface "${SURFACE}"
      --dataset "${DATASET}" --cvfold $CVFOLD --data_path "${DATA_PATH}"
      --save_path "${SAVE_PATH}" --batch_size $BATCH_SIZE --n_workers $N_WORKERS
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUERIES
      --n_episode_test $N_EPISODE_TEST --input_points "${POINTS}"
      --pc_attribs "${PC_ATTRIBS}" --way_pcratio "${WAY_PCRATIO}"
      --way_pcnum "${WAY_PCNUM}"
      --gamma $GAMMA)

CUDA_VISIBLE_DEVICES=$GPU_ID python run_semseg.py "${args[@]}"
