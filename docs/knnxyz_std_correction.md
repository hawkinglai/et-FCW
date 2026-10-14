# knnxyz standard-deviation correction

`pcs_knnxyz` normalizes grouped neighbor offsets by a standard deviation. Release `51686e4` computes it over the **whole batch**; the version that reproduces TMM Table II for ScanObjectNN computes it **per sample**, as `pcs_geobp` does. This note records the change, its provenance, and what it does to every experiment that uses `pcs_knnxyz`. It is a **local correction**, not evidence about which version the published tables used.

## The change

```diff
-        std_xyz = torch.std(grouped_xyz - mean_xyz, keepdim=True)
+        std_xyz = torch.std(grouped_xyz - mean_xyz, dim=[1,2,3], keepdim=True)
```

Applied in all seven `experiments/*/models/surfaces.py` that define `pcs_knnxyz`. Nothing else in model code changes.

### Loader determinism (second change, same PR)

In `ScanObjectNN.py` and `ModelNet.py` (five files), `__getitem__` previously took a **view** of the stored array and shuffled it in place (`np.random.shuffle`), so training items were permuted inside `self.data` with an unseeded global RNG. Now:

```diff
-        pointcloud = self.data[item][:self.num_points]
+        pointcloud = self.data[item][:self.num_points].copy()
...
-            np.random.shuffle(pointcloud)
+            perm = np.random.default_rng([self.seed, int(item)]).permutation(pointcloud.shape[0])
+            pointcloud = pointcloud[perm]
```

`seed` is a new constructor argument, default 3407. The shuffled order depends only on `(seed, item)`. Test items are not shuffled (unchanged). Point order matters because farthest-point sampling starts at index 0.

## Provenance

- Source: VS Code Remote-SSH local history of `reconstruction/models/surfaces.py` (host 192.168.1.239), 50 saved versions kept (2025-07-14 17:53:40 → 2025-12-11 14:05:13).
- `pcs_knnxyz` std argument over time: `dim=-1`/`dim=-2` variants until 2025-07-14 18:02:32; **per-sample `dim=[1,2,3]` from 2025-07-14 18:04:25** through 2025-12-09 15:15:45 (28 saves); **batch-wide from 2025-12-09 15:16:09**, a one-line change (line 186); batch-wide in all later saves. Versions before 2025-07-14 17:53 are not in the history (UNVERIFIED).
- Every release copy (and git `efc07fd`, `d8c36cd`, `b31fdfe`, `51686e4`) carries the batch-wide line; no git blob, archive or agent log contains a per-sample `pcs_knnxyz`.
- Agent involvement: none found. Codex, Claude, Qwen and Hermes transcripts under `~` contain no write to `reconstruction/models/surfaces.py`. Who made the 2025-12-09 edit, and why, is UNVERIFIED; `/mnt/hdd` was not readable.
- On 2025-07-14 18:40 the `s1/s2/s3_scanobjectnn.sh` scripts were set to the C0 configuration with `knnxyz` while the std was per-sample, then reverted at 18:41. Whether anything was run is UNVERIFIED.

## Before / after

Test-selected γ, as historically. "Batch-wide" is release `51686e4`; "per-sample" is this branch (std change **and** loader change). Per-sample runs: torch seed 3407, RTX A5000, Python 3.12.2, torch 2.7.1+cu118; logs in `results/*/rerun_per_sample/`.

| Experiment | Paper | Batch-wide (51686e4) | Per-sample (this branch) |
|---|---:|---:|---:|
| ModelNet40, bz 32, K 66 | 84.8 | 85.1297 (γ 6090) | **85.0486** (γ 5790) |
| ScanObjectNN OBJ-BG, bz 64, K 90 | 66.3 | 53.87 (γ 1032) | **65.92** (γ 4446) |
| ScanObjectNN OBJ-ONLY, bz 64, K 90 | 71.9 | 59.38 (γ 1312) | **72.29** (γ 2276) |
| ScanObjectNN PB-T50-RS, bz 32, K 90 | 55.6 | 63.36 (γ 2752) | **55.14** (γ 3572) |
| ShapeNet-Part, bz 128, γ 230 (mIoU) | 70.50 | 70.18 | **70.35** |

ScanObjectNN gaps to the paper: −0.38 / +0.39 / −0.46 pp, all within ±0.5. The batch-wide ScanObjectNN values are from the forensics runs (`forensics_m1_scan`), not from files in this repository.

ModelNet40 per-sample is **85.0486**, not 84.72. 84.72 was measured with the std change alone on the old loader (in-place unseeded shuffle). The shuffle order alone moves ModelNet40 by 0.33 pp.

### ModelNet-C and ShapeNet-C — the batch-wide version matched these tables exactly

Rows show S0(clean)/S1…S5 as paper-matching batch-wide values versus per-sample values, and the largest absolute difference (pp). Per-sample bz 32 (ModelNet-C), bz 128 (ShapeNet-C).

| ModelNet-C mode / corruption | Batch-wide = paper | Per-sample | max \|Δ\| |
|---|---|---|---:|
| global / jitter | 81.40 / 80.27 / 75.41 / 64.51 / 53.93 / 46.07 | 81.69 / 80.92 / 77.84 / 68.03 / 58.63 / 51.42 | 5.35 |
| local / jitter | 83.39 / 82.70 / 82.05 / 78.48 / 73.14 / 66.65 | 83.31 / 83.06 / 82.29 / 78.69 / 74.11 / 68.23 | 1.58 |
| united / jitter | 84.72 / 84.40 / 82.54 / 78.28 / 69.69 / 60.90 | 84.36 / 84.12 / 82.90 / 78.65 / 72.16 / 65.40 | 4.50 |
| global / add_global | 81.40 / 73.50 / 66.17 / 58.51 / 51.26 / 47.16 | 81.69 / 75.49 / 67.18 / 58.71 / 51.82 / 45.75 | 1.99 |
| local / add_global | 83.39 / 44.69 / 33.43 / 23.58 / 20.14 / 17.42 | 83.31 / 47.29 / 35.17 / 25.53 / 21.64 / 19.29 | 2.60 |
| united / add_global | 84.72 / 58.02 / 45.18 / 36.51 / 29.70 / 24.84 | 84.36 / 59.56 / 48.99 / 41.09 / 34.76 / 29.21 | 5.06 |

| ShapeNet-C | Batch-wide = paper | Per-sample | max \|Δ\| |
|---|---|---|---:|
| knnxyz / jitter | 62.78 / 62.49 / 61.74 / 60.83 / 59.34 / 57.57 | 64.00 / 63.79 / 62.90 / 61.91 / 60.26 / 58.35 | 1.30 |
| knnxyz / add_global | 62.78 / 61.63 / 60.67 / 59.74 / 59.02 / 58.20 | 64.00 / 62.56 / 61.35 / 60.23 / 59.34 / 58.43 | 1.22 |

The batch-wide version reproduced all 30 ModelNet-C and 10 ShapeNet-C corrupted values to two decimals, and the per-sample version does not. The ScanObjectNN Table II values show the opposite. So the published tables are consistent with **different** `pcs_knnxyz` versions per table. Which version produced which table is UNVERIFIED.

## Order dependence of the batch-wide version

All ScanObjectNN h5 files are sorted by class and the loaders use `shuffle=False`, so each batch is nearly single-class (PB-T50-RS test, bz 32: 84.6% of batches). With the batch-wide std the features depend on batch composition. Reordering train and test items (seeds 1, 2) gave, at the C0 configuration:

| Order | OBJ-BG | OBJ-ONLY | PB-T50-RS |
|---|---:|---:|---:|
| file order | 53.87 | 59.38 | 63.36 |
| seed 1 | 56.11 | 59.55 | 43.51 |
| seed 2 | 52.67 | 58.52 | 43.55 |

PB-T50-RS falls to about 43%. The inverted ranking (PB-T50-RS above OBJ-BG) of the batch-wide numbers is an artifact of file order.

## Batch independence (per-sample)

| Run | bz 1 | bz 32 | bz 64 / released |
|---|---|---|---|
| ModelNet40 | 85.0486 (γ 5792) | 85.0486 (γ 5790) | – |
| ScanObjectNN OBJ-BG | 65.9208 (γ 4448) | 65.9208 (γ 4446) | 65.9208 (γ 4446) |
| ScanObjectNN OBJ-ONLY | 72.2892 (γ 2276) | 72.2892 (γ 2276) | 72.2892 (γ 2276) |
| ScanObjectNN PB-T50-RS | 55.1353 (γ 3572) | 55.1353 (γ 3572) | 55.1353 (γ 3572) |

OA is identical in every cell. **γ is not**: ModelNet40 and OBJ-BG differ by one grid step (2) between bz 1 and bz 32. Earlier forensics showed test features at different batch sizes agree to about 1e-6; a float-level reordering of tied accuracies would explain the shift, but this was not checked (UNVERIFIED).

Other bz = 1 checks were not required and were run only where listed here: ShapeNet-Part mIoU 70.35 / 70.21 / 70.26 at bz 128 / 32 / 1; ModelNet-C jitter agrees across bz 1 and 32 within 0.04 pp, and add_global differs by up to 1.33 pp; ShapeNet-C jitter and add_global agree across bz 128 / 32 within 0.14 pp. The segmentation and corruption loaders resample with an unseeded NumPy generator whose per-worker stream depends on batch layout. That is the likely cause and was not tested (UNVERIFIED); those loaders were out of scope for the seeding change.

## What was and was not re-run

| Experiment | Uses `pcs_knnxyz` | Status |
|---|---|---|
| ModelNet40 (`modelnet40`) | yes | re-run, bz 32 and 1 |
| ScanObjectNN xyz (`modelnet40/run_scanobjectnn.sh`) | yes | re-run, bz released, 1 and 32 |
| ShapeNet-Part (`shapenet_part`) | yes | re-run, bz 128 (also 32 and 1) |
| ModelNet-C (`modelnetc`) | yes | re-run, bz 32 (also 1) |
| ShapeNet-C (`shapenetc`) | yes (knnxyz row; the runner also evaluates geobp) | re-run, bz 128 and 32 (jitter also bz 1) |
| RISP rotation (`risp_rotation`) | no: only `tFCW_encoding` is imported; `pcs_knnxyz` appears only in a demo `__main__` | not re-run; numbers unchanged |
| GeoPCSD ScanObjectNN / ShapeNet-Part | no (`geobp`) | source of `pcs_knnxyz` patched but not on the run path. ScanObjectNN loader changed, see below |
| S3DIS few-shot | yes, outside this release | **not re-run** (deferred) |

### GeoPCSD ScanObjectNN loader note

The loader change reaches the GeoPCSD ScanObjectNN experiment. Re-run at its recorded settings: 69.5353 / 72.9776 / 57.4948, against the recorded 69.7074 / 72.9776 / 58.0500. The recorded values came from the old in-place unseeded shuffle. Its `summary.json` is left unchanged. To reproduce the recorded values, use the `ScanObjectNN.py` of `51686e4`.

## Reproducing the batch-wide results

Revert the one `std_xyz` line (and, for byte-exact old shuffle behavior, the loader change) as in `51686e4`. The legacy values remain in the `summary.json` files under `legacy_batch_wide_std`; the ModelNet40 85.1297 is labelled "batch-wide std regression".
