# S3DIS few-shot semantic segmentation (TMM Table III)

Ported from local source (`reconstruction/`) with hyperparameters unchanged. Corresponds to IEEE TMM Table III (`t-FCW` via `--surface knnxyz` and `t-FCW w/ GeoPCSD` via `--surface geobp`).

## Running

Set `TFCW_S3DIS_ROOT` to the S3DIS `blocks_bs1_s1/` directory (containing `data/*.npy` and a sibling `../meta/s3dis_classnames.txt` or `../../meta/s3dis_classnames.txt`), then run from the repository root:

```bash
TFCW_S3DIS_ROOT=/path/to/S3DIS/blocks_bs1_s1 bash experiments/s3dis_fewshot/run.sh
```

Override fold, shot, surface, point schedule, or gamma via environment variables (e.g., `CVFOLD=0 K_SHOT=1 SURFACE=geobp GAMMA=1`, or `SURFACE=knnxyz GAMMA=31`; for 2048-point episodes as in Seg-NN, pass `POINTS="[2048,1024,512,256]"`).

## Notes on preserved behaviors & fixes

- **Canonical model (`models/sem2.py`):** `run_semseg.py:11` imports `models/sem2.py` (which omits the redundant `attention_rgb` branch and offloads branch outputs to CPU). `models/sem2.py` outperformed `models/sem.py` on all 8 validation cells (`+0.07` to `+1.36` pp); `models/sem.py` has been removed from this final branch (retained untouched in `add/local-code`).
- **GeoPCSD neighbor index fix (`[1] -> [2]`):** Fixed the duplicate `neighbor_2nd` index in `models/semsurfaces.py:119` (`index=[2]`, matching `experiments/geopcsd_{scanobjectnn,shapenet_part}/models/surfaces.py:119`).
- **Stage-0 ordering (A03):** Fixed in `models/semsurfaces.py` (`pcs_geobp` and `pcs_knnxyz`) so stage-0 grouping centers keep input point order (`xyz_list[0]`) while stages 1–4 remain identical to the unfixed code.
- **Dataloader (Seg-NN origin):** `dataloaders/loader.py:41-52` retains Seg-NN / attMPTI's label-dependent foreground stratification (`random_sample=False`); `dataloaders/s3dis.py` resolves `meta/s3dis_classnames.txt` via `TFCW_S3DIS_CLASSNAMES` / relative paths with the original path as fallback.
- **Cross-block normalization:** `models/semsurfaces.py:156,173,204,221` retain batch-wide `torch.std` across all support (`N_way * K_shot`) or query (`N_way * n_queries`) blocks in an episode.

## Verification results (`sem2.py`, 2-way, 1,500 episodes, Seg-NN protocol)

### 2,048 points (aligned with Seg-NN / COSeg)

| Surface (γ) | Split / Shot | Pre-fix `sem2` (`add/local-code`, s0) | Fixed V0 (`ready/icml-audit`, s0) | Fixed V0 (3-seed mean ± std) | Fixed V2 per-block (3-seed mean ± std) | TMM Table III (1,024 pts) |
|---|---|---:|---:|---:|---:|---:|
| `geobp` (γ=1) | S0 1-shot | 45.59 | 45.07 | 44.93 ± 0.10 | 44.34 ± 0.17 | 45.86 |
| `geobp` (γ=1) | S0 5-shot | 53.86 | 53.27 | 53.44 ± 0.22 | 53.07 ± 0.25 | 55.46 |
| `geobp` (γ=1) | S1 1-shot | 45.27 | 44.83 | 44.78 ± 0.11 | 45.13 ± 0.35 | 44.23 |
| `geobp` (γ=1) | S1 5-shot | 55.62 | 54.89 | 54.61 ± 0.23 | 55.20 ± 0.19 | 54.45 |
| `knnxyz` (γ=31) | S0 1-shot | 42.93 | 42.83 | 43.02 ± 0.22 | 42.33 ± 0.01 | 43.42 |
| `knnxyz` (γ=31) | S0 5-shot | 51.97 | 51.52 | 51.86 ± 0.24 | 51.46 ± 0.26 | 52.50 |
| `knnxyz` (γ=31) | S1 1-shot | 45.02 | 45.12 | 45.48 ± 0.26 | 45.93 ± 0.24 | 43.54 |
| `knnxyz` (γ=31) | S1 5-shot | 55.99 | 55.68 | 55.50 ± 0.40 | 56.04 ± 0.26 | 53.84 |

### 1,024 points (`s3dis.sh` point schedule, seed 0)

| Surface (γ) | Split / Shot | Pre-fix `sem2` (`idx2=1`, pre-A03) | Fixed `sem2` V0 (`idx2=2`, post-A03) | TMM Table III |
|---|---|---:|---:|---:|
| `geobp` (γ=1) | S0 1-shot | 46.65 | 46.30 | 45.86 |
| `geobp` (γ=1) | S0 5-shot | 54.59 | 54.23 | 55.46 |
| `geobp` (γ=1) | S1 1-shot | 45.75 | 45.34 | 44.23 |
| `geobp` (γ=1) | S1 5-shot | 56.48 | 55.97 | 54.45 |
| `knnxyz` (γ=31) | S0 1-shot / S0 5-shot / S1 1-shot / S1 5-shot | 42.62 / 51.79 / 44.84 / 55.58 | — | 43.42 / 52.50 / 43.54 / 53.84 |
