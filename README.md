# ET-FCW: Empowered t-FCW Graph Representation for Point Cloud Analysis

[![arXiv](https://img.shields.io/badge/arXiv-2605.15475-b31b1b.svg)](https://arxiv.org/abs/2605.15475)

Official implementation of *"A Unified Non-Parametric and Interpretable Point Cloud Analysis via t-FCW Graph Representation"* (IEEE TMM 2026).

## Overview

ET-FCW is a non-parametric, training-free framework for point cloud classification, part segmentation, and robustness evaluation. It translates point clouds into compact descriptor-based graphs via **transposed Fully-Connected Weighted (t-FCW)** representations, enabling interpretable predictions through similarity-based memory banks.

- Zero trainable parameters
- ~7 seconds for ModelNet40 on an RTX A5000
- Works standalone or as a plug-in module

## Quick Start

```bash
pip install -r requirements.txt
```

Datasets are expected under `data/` (symlink or copy):

```
data/
├── modelnet40_ply_hdf5_2048/   # ModelNet40
├── shapenetcore_partanno_segmentation_benchmark_v0_normal/  # ShapeNet Part
├── modelnet_c/                  # ModelNet40-C
└── shapenet_c/                  # ShapeNet-C
```

Set `DATASET_ROOT` to override the default `data/` path.

## Experiments

### ModelNet40 Classification

```bash
bash scripts/run_cls.sh
```

| Method | Accuracy | Speed (samples/s) |
|--------|----------|-------------------|
| t-FCW  | 84.8%    | 1727              |

### ShapeNet Part Segmentation

```bash
bash scripts/run_part.sh
```

| Method | mIoU |
|--------|------|
| t-FCW  | 70.4% |

### ModelNet40-C Robustness

```bash
bash scripts/run_modelnetc.sh
```

## Structure

```
├── models/          # t-FCW encoder, PCSD formatters, classification/segmentation nets
├── dataloaders/     # ModelNet40, ShapeNet Part, ModelNet40-C, ShapeNet-C
├── scripts/         # Shell wrappers for each experiment
├── run_cls.py       # ModelNet40 classification
├── run_part.py      # ShapeNet part segmentation (mIoU)
├── run_part2.py     # ShapeNet part segmentation (per-category)
├── run_modelnetc.py # ModelNet40-C corruption evaluation
└── requirements.txt
```

## Citation

```bibtex
@ARTICLE{11617202,
  author={Lai, Haijian and Liu, Bowen and Xu, Man and Lam, Chan-Tong and Macedo, Jo{\~a}o and Ng, Benjamin and Im, Sio-Kei},
  journal={IEEE Transactions on Multimedia}, 
  title={A Unified Non-Parametric and Interpretable Point Cloud Analysis Via t-FCW Graph Representation}, 
  year={2026},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TMM.2026.3716075}
}
```

## Data Preparation

| Dataset | Source |
|---------|--------|
| ModelNet40, ShapeNet Part | [Point-NN](https://github.com/ZrrSkywalker/Point-NN) |
| ModelNet40-C, ShapeNet-C | [PointCloud-C](https://github.com/ldkong1205/PointCloud-C) |

Place under `data/` as described above.
