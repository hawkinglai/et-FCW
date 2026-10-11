# Evaluation protocols

| Experiment | γ protocol | Result record |
|---|---|---|
| ModelNet40 | Best labeled-test accuracy, grid 0,2,…,9998; first maximum | [Accepted](../results/modelnet40/port_check.json) |
| ShapeNet-Part | Fixed γ230 | [Accepted](../results/shapenet_part/port_check.json) |
| RISP rotation | Independent labeled-test selection for each condition, grid 0,2,…,9998 | [Accepted with installed PyTorch3D](../results/risp_rotation/port_check_installed.json) |
| ModelNet-C | Independent labeled-test selection for clean and each severity, grid 0,10,…,9990; runner does not print γ | [Accepted](../results/modelnetc/port_check.json) |
| ShapeNet-C | Fixed γ230; ten knnxyz corruption values | [Accepted](../results/shapenetc/port_check.json) |
| Corrected GeoPCSD ScanObjectNN | Best labeled-test accuracy, grid 0,2,…,9998 | [Saved corrected results](../results/geopcsd_scanobjectnn/summary.json) |
| Corrected GeoPCSD ShapeNet-Part | Fixed γ230 | [Saved corrected results](../results/geopcsd_shapenet_part/summary.json) |

All five plain/RISP experiments passed numerical acceptance. GeoPCSD records retain corrected local results; their current checks compare saved logs with summaries and verify packaging, without a new numerical execution.

`pcs_knnxyz` standard deviation spans the whole batch: preserve batch size and composition. ShapeNet-Part and ShapeNet-C retain NumPy replacement sampling without an explicit top-level NumPy seed. ModelNet-C resamples non-1024 clouds from the entire cloud, including appended points. Keep the original file ordering, DataLoader workers and Torch seed 3407.

RISP ignores surface and rescale, rotates xyz and normals together, and evaluates all four conditions internally. Its train loader drops the last batch; historical FPS caches are required for the recorded inputs. PyTorch3D 0.7.8 is installed as a package.

Corruption filenames indexed 0–4 correspond to severities 1–5. `add_global` corresponds numerically to paper Global. ShapeNet-C test category labels select the part memory bank.

The [separate train-selected ModelNet-C experiment](../../mnc_fixedgamma_v1/summary.md) uses saved batch-32 features, five-fold stratified clean-training CV (seed 0), γ grid 0,10,…,9990, and the smallest γ on ties. One γ per mode is fixed for clean and all ten corruptions. NEW EXPERIMENT — NOT A REPRODUCTION.

The [GeoPCSD correction](geopcsd_correction.md) and [patch](../provenance/geopcsd_neighbor_fix.patch) record the neighbor-index change in the two corrected experiments.
