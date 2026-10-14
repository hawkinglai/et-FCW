# knnxyz standard-deviation correction

`pcs_knnxyz` normalizes grouped neighbor offsets by a standard deviation. Release `51686e4` computes it over the **whole batch**; the version that reproduces TMM Table II for ScanObjectNN computes it **per sample**, as `pcs_geobp` does. This note records the change, its provenance, and what it does to every experiment that uses `pcs_knnxyz`. Table II is consistent with the per-sample version and Tables VII/VIII with the batch-wide version; this is not directly verified.

## The change

```diff
-        std_xyz = torch.std(grouped_xyz - mean_xyz, keepdim=True)
+        std_xyz = torch.std(grouped_xyz - mean_xyz, dim=[1,2,3], keepdim=True)
```

Applied in all seven `experiments/*/models/surfaces.py` that define `pcs_knnxyz`. Nothing else in model code changes.

### Loader determinism (second change, same PR)

In `modelnet40/dataloaders/{ScanObjectNN,ModelNet}.py` and `modelnetc/dataloaders/ModelNet.py` (three files), `__getitem__` previously took a **view** of the stored array and shuffled it in place (`np.random.shuffle`), so training items were permuted inside `self.data` with an unseeded global RNG. Now:

```diff
-        pointcloud = self.data[item][:self.num_points]
+        pointcloud = self.data[item][:self.num_points].copy()
...
-            np.random.shuffle(pointcloud)
+            perm = np.random.default_rng([self.seed, int(item)]).permutation(pointcloud.shape[0])
+            pointcloud = pointcloud[perm]
```

`seed` is a new constructor argument, default 3407. The loaders of `experiments/geopcsd_scanobjectnn` were intentionally left unchanged: that experiment uses `geobp`, which never used the batch-wide std. The shuffled order depends only on `(seed, item)`. Test items are not shuffled (unchanged). Point order matters because farthest-point sampling starts at index 0.

## Provenance

- Source: the editor's local file history of `reconstruction/models/surfaces.py`, 50 saved versions kept (2025-07-14 17:53:40 → 2025-12-11 14:05:13).
- `pcs_knnxyz` std argument over time: `dim=-1`/`dim=-2` variants until 2025-07-14 18:02:32; **per-sample `dim=[1,2,3]` from 2025-07-14 18:04:25** through 2025-12-09 15:15:45 (28 saves); **batch-wide from 2025-12-09 15:16:09**, a one-line change (line 186); batch-wide in all later saves. Versions before 2025-07-14 17:53 are not in the history (UNVERIFIED).
- Every release copy (and git `efc07fd`, `d8c36cd`, `b31fdfe`, `51686e4`) carries the batch-wide line; no git blob, archive or agent log contains a per-sample `pcs_knnxyz`.
- Agent involvement: none found. Codex, Claude, Qwen and Hermes transcripts under `~` contain no write to `reconstruction/models/surfaces.py`. Who made the 2025-12-09 edit, and why, is UNVERIFIED; one mounted volume was not readable.
- On 2025-07-14 18:40 the `s1/s2/s3_scanobjectnn.sh` scripts were set to the C0 configuration with `knnxyz` while the std was per-sample, then reverted at 18:41. Whether anything was run is UNVERIFIED.

## Before / after

Test-selected γ, as historically. "Batch-wide" is release `51686e4`; "per-sample" is this branch (std change **and** loader change; GeoPCSD ScanObjectNN unchanged). Per-sample runs: torch seed 3407, RTX A5000, Python 3.12.2, torch 2.7.1+cu118; logs in `results/*/rerun_per_sample/` (local absolute paths replaced by `$REPO`, `$DATA_ROOT`, `$PYTHON`).

| Experiment | Paper | Batch-wide (51686e4) | Per-sample (this branch) |
|---|---:|---:|---:|
| ModelNet40, bz 32, K 66 | 84.8 | 85.1297 (γ 6090) | **85.0486** (γ 5790) |
| ScanObjectNN OBJ-BG, bz 64, K 90 | 66.3 | 53.87 (γ 1032) | **65.92** (γ 4446) |
| ScanObjectNN OBJ-ONLY, bz 64, K 90 | 71.9 | 59.38 (γ 1312) | **72.29** (γ 2276) |
| ScanObjectNN PB-T50-RS, bz 32, K 90 | 55.6 | 63.36 (γ 2752) | **55.14** (γ 3572) |
| ShapeNet-Part, bz 128, γ 230 (mIoU) | 70.50 | 70.18 | **70.35** |

ScanObjectNN gaps to the paper: −0.38 / +0.39 / −0.46 pp, all within ±0.5. The batch-wide ScanObjectNN values are from the forensics runs , not from files in this repository.

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

## Is the batch-wide version order-dependent on ModelNet40 / ModelNet-C / ShapeNet-C?

Diagnostic on the batch-wide code of `51686e4` (scratch copies; no code change). Raw logs: `results/order_diag/`.

### (1) Are the files grouped by class?

Labels are read in the loaders' own order (ModelNet40: `glob` order of `ply_data_{test,train}*.h5`, concatenated). "Runs" is the number of maximal same-label runs; "random" is its expectation for a random order. Single-class batch fraction is at the released batch size (ModelNet40 and ModelNet-C 32, ShapeNet-C 128).

| file | n | classes | sorted by label | label runs | max run | bz | single-class batches | mean majority fraction |
|---|---:|---:|---|---|---:|---:|---:|---:|
| `modelnet40_test` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.123 |
| `modelnet40_train` | 9840 | 40 | no | 9442 (random: 9449) | 3 | 32 | 0.0000 | 0.132 |
| `modelnet_c/clean` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.119 |
| `modelnet_c/jitter_0` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.119 |
| `modelnet_c/jitter_1` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.119 |
| `modelnet_c/jitter_2` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.119 |
| `modelnet_c/jitter_3` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.119 |
| `modelnet_c/jitter_4` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.119 |
| `modelnet_c/add_global_0` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.119 |
| `modelnet_c/add_global_1` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.119 |
| `modelnet_c/add_global_2` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.119 |
| `modelnet_c/add_global_3` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.119 |
| `modelnet_c/add_global_4` | 2468 | 40 | no | 2382 (random: 2382) | 3 | 32 | 0.0000 | 0.119 |
| `shapenet_c/clean` | 2874 | 16 | no | 2346 (random: 2358) | 6 | 128 | 0.0000 | 0.306 |
| `shapenet_c/jitter_0` | 2874 | 16 | no | 2346 (random: 2358) | 6 | 128 | 0.0000 | 0.306 |
| `shapenet_c/jitter_1` | 2874 | 16 | no | 2346 (random: 2358) | 6 | 128 | 0.0000 | 0.306 |
| `shapenet_c/jitter_2` | 2874 | 16 | no | 2346 (random: 2358) | 6 | 128 | 0.0000 | 0.306 |
| `shapenet_c/jitter_3` | 2874 | 16 | no | 2346 (random: 2358) | 6 | 128 | 0.0000 | 0.306 |
| `shapenet_c/jitter_4` | 2874 | 16 | no | 2346 (random: 2358) | 6 | 128 | 0.0000 | 0.306 |
| `shapenet_c/add_global_0` | 2874 | 16 | no | 2346 (random: 2358) | 6 | 128 | 0.0000 | 0.306 |
| `shapenet_c/add_global_1` | 2874 | 16 | no | 2346 (random: 2358) | 6 | 128 | 0.0000 | 0.306 |
| `shapenet_c/add_global_2` | 2874 | 16 | no | 2346 (random: 2358) | 6 | 128 | 0.0000 | 0.306 |
| `shapenet_c/add_global_3` | 2874 | 16 | no | 2346 (random: 2358) | 6 | 128 | 0.0000 | 0.306 |
| `shapenet_c/add_global_4` | 2874 | 16 | no | 2346 (random: 2358) | 6 | 128 | 0.0000 | 0.306 |

None of the files is sorted or grouped by class: run counts equal the random expectation (within 12 of it), the longest run is 3 (ModelNet) or 6 (ShapeNet-C), and no batch is single-class. For comparison, the ScanObjectNN files are sorted into 15 runs (see above). ShapeNet-C label here is the shape category (`label`); `pid` was not used.

### (2) Random test/train order, batch-wide version

Seeds 1–3: a seeded permutation applied to the train and the test loaders through an external wrapper (`DataLoader` datasets wrapped in `Subset`; runner source unchanged). The "original" column is the file order. Because the files are not class-grouped, ShapeNet-C was **not** order-tested (condition for running it was that the files be class-grouped; they are not).

**ModelNet40** (bz 32, K 66, metric 2, rescale 0.8):

| order | OA | γ |
|---|---:|---:|
| original | 85.1297 | 6090 |
| seed 1 | 85.7374 | 6792 |
| seed 2 | 84.8460 | 8616 |
| seed 3 | 85.2917 | 8140 |

Seeds 1–3: min 84.8460, max 85.7374, mean 85.2917; largest |Δ| from the original 0.61 pp.

**ModelNet-C** (bz 32, K 110, metric 2, rescale 0.8; selected γ per set). "orig" is the file order; min/max/mean are over seeds 1–3; last column is the largest |Δ| from orig over the three seeds. The three orig rows for jitter equal the recorded batch-wide values.

| corruption | mode | set | orig | min | max | mean | max \|Δ\| |
|---|---|---|---:|---:|---:|---:|---:|
| jitter | global | clean | 81.40 | 81.00 | 81.40 | 81.19 | 0.40 |
| jitter | global | S1 | 80.27 | 79.62 | 80.59 | 80.17 | 0.65 |
| jitter | global | S2 | 75.41 | 75.12 | 76.22 | 75.51 | 0.81 |
| jitter | global | S3 | 64.51 | 63.94 | 64.75 | 64.39 | 0.57 |
| jitter | global | S4 | 53.93 | 53.61 | 53.77 | 53.69 | 0.32 |
| jitter | global | S5 | 46.07 | 45.62 | 46.64 | 46.15 | 0.57 |
| jitter | local | clean | 83.39 | 83.02 | 83.31 | 83.16 | 0.37 |
| jitter | local | S1 | 82.70 | 82.58 | 83.10 | 82.77 | 0.40 |
| jitter | local | S2 | 82.05 | 81.81 | 82.21 | 82.02 | 0.24 |
| jitter | local | S3 | 78.48 | 77.88 | 79.05 | 78.51 | 0.60 |
| jitter | local | S4 | 73.14 | 72.61 | 73.01 | 72.81 | 0.53 |
| jitter | local | S5 | 66.65 | 66.86 | 67.26 | 67.11 | 0.61 |
| jitter | united | clean | 84.72 | 85.05 | 85.17 | 85.12 | 0.45 |
| jitter | united | S1 | 84.40 | 84.32 | 84.68 | 84.52 | 0.28 |
| jitter | united | S2 | 82.54 | 82.82 | 83.10 | 82.99 | 0.56 |
| jitter | united | S3 | 78.28 | 77.88 | 78.04 | 77.97 | 0.40 |
| jitter | united | S4 | 69.69 | 69.57 | 69.89 | 69.78 | 0.20 |
| jitter | united | S5 | 60.90 | 60.98 | 62.16 | 61.48 | 1.26 |
| add_global | global | clean | 81.40 | 81.00 | 81.40 | 81.19 | 0.40 |
| add_global | global | S1 | 73.50 | 74.11 | 74.68 | 74.41 | 1.18 |
| add_global | global | S2 | 66.17 | 65.19 | 66.00 | 65.54 | 0.98 |
| add_global | global | S3 | 58.51 | 57.74 | 58.27 | 58.09 | 0.77 |
| add_global | global | S4 | 51.26 | 50.97 | 51.66 | 51.38 | 0.40 |
| add_global | global | S5 | 47.16 | 45.87 | 46.43 | 46.11 | 1.29 |
| add_global | local | clean | 83.39 | 83.02 | 83.31 | 83.16 | 0.37 |
| add_global | local | S1 | 44.69 | 44.29 | 45.50 | 44.89 | 0.81 |
| add_global | local | S2 | 33.43 | 31.12 | 33.83 | 32.58 | 2.31 |
| add_global | local | S3 | 23.58 | 23.30 | 24.47 | 23.81 | 0.89 |
| add_global | local | S4 | 20.14 | 19.25 | 20.30 | 19.86 | 0.89 |
| add_global | local | S5 | 17.42 | 16.49 | 17.67 | 16.91 | 0.93 |
| add_global | united | clean | 84.72 | 85.05 | 85.17 | 85.12 | 0.45 |
| add_global | united | S1 | 58.02 | 57.46 | 59.24 | 58.16 | 1.22 |
| add_global | united | S2 | 45.18 | 45.62 | 46.35 | 46.03 | 1.17 |
| add_global | united | S3 | 36.51 | 35.78 | 37.97 | 36.86 | 1.46 |
| add_global | united | S4 | 29.70 | 29.78 | 30.83 | 30.25 | 1.13 |
| add_global | united | S5 | 24.84 | 25.12 | 25.28 | 25.21 | 0.44 |

Largest |Δ| from the original order: 1.26 pp (jitter) and 2.31 pp (add_global). The jitter files have 1024 points. The `add_global` files have 1034–1074 points and are resampled to 1024 with an unseeded NumPy generator whose per-worker stream depends on item-to-worker assignment, so changing the order also changes the resampled points for `add_global`; the two effects are not separated here. The jitter rows are free of that confound. For comparison, PB-T50-RS fell from 63.36 to about 43 under reordering (section above).

## Batch independence (per-sample)

| Run | bz 1 | bz 32 | bz 64 / released |
|---|---|---|---|
| ModelNet40 | 85.0486 (γ 5792) | 85.0486 (γ 5790) | – |
| ScanObjectNN OBJ-BG | 65.9208 (γ 4448) | 65.9208 (γ 4446) | 65.9208 (γ 4446) |
| ScanObjectNN OBJ-ONLY | 72.2892 (γ 2276) | 72.2892 (γ 2276) | 72.2892 (γ 2276) |
| ScanObjectNN PB-T50-RS | 55.1353 (γ 3572) | 55.1353 (γ 3572) | 55.1353 (γ 3572) |

OA identical; γ may differ by one grid step (float ties): ModelNet40 and OBJ-BG differ by one grid step (2) between bz 1 and bz 32. Test features at different batch sizes agree to about 1e-6 in the forensics runs. The tie explanation itself was not checked.

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
| GeoPCSD ScanObjectNN / ShapeNet-Part | no (`geobp`) | source of `pcs_knnxyz` patched but not on the run path. loader intentionally unchanged |
| S3DIS few-shot | yes, outside this release | **not re-run** (deferred) |

## Reproducing the batch-wide results

Revert the one `std_xyz` line (and, for byte-exact old shuffle behavior, the loader change) as in `51686e4`. The legacy values remain in the `summary.json` files under `legacy_batch_wide_std`; the ModelNet40 85.1297 is labelled "batch-wide std regression".
