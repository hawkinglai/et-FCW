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
- **Stage-0 ordering (A03):** Fixed in `models/semsurfaces.py` (`pcs_geobp` and `pcs_knnxyz`) on `ready/icml-audit` so stage-0 grouping centers keep input point order (`xyz_list[0]`) while stages 1–4 remain identical to the unfixed code.

## Verification results (2-way, 2048 pts, 1,500 episodes, Seg-NN protocol)

| Surface (γ) | Split / Shot | Pre-A03 (`add/local-code`, s0) | Post-A03 V0 (`ready/icml-audit`, s0) | Post-A03 V0 (3-seed mean ± std) | TMM Table III |
|---|---|---:|---:|---:|---:|
| `geobp` (γ=1) | S0 1-shot | 44.72 | 44.30 | 44.24 ± 0.05 | 45.86 |
| `geobp` (γ=1) | S0 5-shot | 52.66 | 51.96 | 52.05 ± 0.15 | 55.46 |
| `geobp` (γ=1) | S1 1-shot | 43.86 | 43.48 | 43.32 ± 0.11 | 44.23 |
| `geobp` (γ=1) | S1 5-shot | 54.37 | 53.71 | 53.38 ± 0.29 | 54.45 |
| `knnxyz` (γ=31) | S0 1-shot | 43.09 | 42.98 | 43.12 ± 0.23 | 43.42 |
| `knnxyz` (γ=31) | S0 5-shot | 51.83 | 51.27 | 51.56 ± 0.22 | 52.50 |
| `knnxyz` (γ=31) | S1 1-shot | 43.59 | 43.73 | 44.04 ± 0.23 | 43.54 |
| `knnxyz` (γ=31) | S1 5-shot | 55.33 | 54.96 | 54.72 ± 0.39 | 53.84 |
