# Protocol and command notes

## Faithful historical reproduction

The following entry points are **references to the external historical companion source**, not executable files shipped here:

| Result | Historical entry point | Release evidence |
|---|---|---|
| ModelNet40 plain | `reconstruction/scripts/modelnet.sh` from the parent of `reconstruction/` | [summary](../results/modelnet40/summary.json), [log](../results/modelnet40/stdout_stderr.log) |
| ShapeNet-Part plain | `reconstruction/scripts/shapenet.sh` | [summary](../results/shapenet_part/summary.json), [log](../results/shapenet_part/stdout_stderr.log) |
| RISP rotation | `reconstruction/run_cls_ri.py` with the arguments recorded in the [experiment note](../experiments/risp_rotation/README.md) | [summary](../results/risp_rotation/summary.json), [log](../results/risp_rotation/stdout_stderr.log) |
| Table VII plain | `reconstruction/run_cls_modelnetcv2.py`; evaluated through a scope-limited original-compatible copy | [summary](../results/modelnetc/summary.json), [logs](../results/modelnetc/) |
| Table VIII plain | `reconstruction/run_part_cv2.py`; evaluated through a scope-limited original-compatible copy | [summary](../results/shapenetc/summary.json), [logs](../results/shapenetc/) |

ModelNet40 classification searched γ=0,2,…,9998 on labeled test data. RISP selected γ independently for each rotated test condition. ModelNet-C selected γ independently on each corrupted severity's test labels. Those choices are historical behaviors and introduce test-set selection bias. ShapeNet-Part used fixed γ230. The original ShapeNet-Part loader sampled points with replacement without an explicit top-level NumPy seed. ModelNet-C sampled from the full corrupted cloud, including appended points.

The RISP shell script included unsupported rotation arguments; the saved execution omitted only those invalid arguments while the original runner executed all four rotations internally and rotated xyz plus normals. Table VII and VIII release summaries preserve only plain results; corrected GeoPCSD outputs are separate reconstructions.

## Corrected GeoPCSD reconstruction

- [ScanObjectNN launcher](../experiments/geopcsd_scanobjectnn/run.sh) and [source](../experiments/geopcsd_scanobjectnn/run_cls.py).
- [ShapeNet-Part launcher](../experiments/geopcsd_shapenet_part/run.sh) and [source](../experiments/geopcsd_shapenet_part/run_part.py).

The release copies apply only the recorded neighbor-index correction plus **path-only** environment-variable substitutions in loaders. The latter require the caller to supply the same datasets at explicit paths; no sampling, model, γ selection, or voting equation was changed. These source copies have unresolved redistribution rights, as described in [LICENSE_STATUS.md](../LICENSE_STATUS.md).

## External source and paths

The authoritative source is expected as a sibling `reconstruction/` directory in a separately licensed companion archive. Original summaries had absolute developer paths; curated public summaries retain experiment settings and numbers without those machine-local paths. The complete original command strings remain in the private forensic audit, not in this source-only public tree. This release is therefore **not source-self-contained for faithful historical runs**.
