# Local GeoPCSD correction

A duplicate-neighbor transcription error was introduced during local reconstruction/copying of GeoPCSD source. This is **not evidence that the published paper contained the error**. The sole intended algorithmic intervention in the paired comparison was:

```diff
- neighbor_2nd = neighbor[..., 1, :]
+ neighbor_2nd = neighbor[..., 2, :]
```

The actual source-level [patch](../provenance/geopcsd_neighbor_fix.patch) changes the corresponding `torch.index_select` index. When the first and second selected neighbors are identical, `edge1 == edge2`, so the cross product used for the local normal degenerates to zero. The canonical source shipped here selects index 2 for the second neighbor; both public experiments use the same corrected surface file.

| Dataset | Before local correction | Corrected | Gain |
|---|---:|---:|---:|
| ScanObjectNN OBJ-BG | 69.0189 | 69.7074 | +0.6885 pp |
| ScanObjectNN OBJ-ONLY | 71.7728 | 72.9776 | +1.2048 pp |
| ScanObjectNN PB-T50-RS | 57.8071 | 58.0500 | +0.2429 pp |
| ShapeNet-Part | 68.34 | 68.68 | +0.34 pp |

The correction improved all three evaluated ScanObjectNN splits and ShapeNet-Part segmentation. The [classification](../results/geopcsd_scanobjectnn/summary.json) and [segmentation](../results/geopcsd_shapenet_part/summary.json) summaries retain the measured values. Complete paired source and logs remain in the separate forensic workspace. Corrected outcomes are labeled **CORRECTED_RECONSTRUCTION**, not faithful verification of unpublished GeoPCSD source.
