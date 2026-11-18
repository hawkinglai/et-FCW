# S3DIS few-shot semantic segmentation (TMM Table III)

Ported from local source (`reconstruction/`) with hyperparameters unchanged. Corresponds to IEEE TMM Table III (`t-FCW` via `--surface knnxyz` and `t-FCW w/ GeoPCSD` via `--surface geobp`).

## Running

Set `TFCW_S3DIS_ROOT` to the S3DIS `blocks_bs1_s1/` directory (containing `data/*.npy` and a sibling `../meta/s3dis_classnames.txt` or `../../meta/s3dis_classnames.txt`), then run from the repository root:

```bash
TFCW_S3DIS_ROOT=/path/to/S3DIS/blocks_bs1_s1 bash experiments/s3dis_fewshot/run.sh
```

Override fold, shot, surface, point schedule, or gamma via environment variables (e.g., `CVFOLD=0 K_SHOT=1 SURFACE=geobp GAMMA=1`, or `SURFACE=knnxyz GAMMA=31`; for 2048-point episodes as in Seg-NN, pass `POINTS="[2048,1024,512,256]"`).

## Notes on preserved behaviors

- **Files:** `run_semseg.py`, `models/sem.py`, `models/sem2.py`, `models/semsurfaces.py`, `dataloaders/loader.py`, `dataloaders/scannet.py` are copied verbatim from local `reconstruction/`; `dataloaders/s3dis.py` differs only in resolving `meta/s3dis_classnames.txt` relative to `data_path` (or `TFCW_S3DIS_CLASSNAMES`) with the original relative path as fallback.
- **Dataloader (Seg-NN origin):** `dataloaders/loader.py:41-52` samples support and query block points using label-dependent foreground stratification (the foreground-leakage sampler inherited from Seg-NN / attMPTI).
- **Cross-block normalization:** `models/semsurfaces.py:156,173,203,219` compute `torch.std` over the entire episode batch tensor (support `N_way * K_shot` blocks together, query `N_way * n_queries` blocks together) rather than per block.
- **Stage-0 ordering (A03):** In the original `models/sem.py:31-36,111`, `models/sem2.py:31-36,111`, and `models/semsurfaces.py:143,195`, stage 0 runs full-count FPS while `xyz_list[0]` keeps input point order (fixed on the integrated `ready/icml-audit` branch).
